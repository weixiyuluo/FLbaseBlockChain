package main

import (
	"fmt"
	"log"
	"math"

	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"

	"time"

	"github.com/DataDog/go-python3"
)

var (
	Global         []float64     //global model
	g              int           //comm公钥
	p              int           //大素数
	e              int           //精度
	numofAgg       int       = 0 //参与聚合的人数
	numRand        []int         //随机分配噪声接收
	recvedagg      [7850]float64
	iterationCount = -1 //轮数统计
	bc             Blockchain
)

var (
	pyTorchModule *python3.PyObject //加载python文件

	pyInitFunc *python3.PyObject //初始化
	pyPrivFunc *python3.PyObject //计算梯度
	pyNoise    *python3.PyObject //noise
	pyDownload *python3.PyObject //update model by global
	pyUpdate   *python3.PyObject //update by grad
	pyAccur    *python3.PyObject //get test accur
	pyCompare  *python3.PyObject //compare train set accur before update and after update
	pyKrum     *python3.PyObject //krun sorce     //2f+2<n
	pyComm     *python3.PyObject //value of comm
	pyClose    *python3.PyObject //close writer
	piLage     *python3.PyObject
)

/*
type Update struct {
	//SourceID 		int     //trainer id
	Iteration int
	Delta     []float64 //grad
	Noise     []float64
	//Accepted        bool      //是否被验证者接收
	//SignatureList
}




type Honest struct {
	update         Update
	aggregatedGrad []Update //梯度聚合  //存储收到的梯度
	CommOfDelta    int64    // a commitment to delta
	CommOfNoise    int64
	//clientId                int
	//blockUpdates      []Update
	//bc                *Blockchain      //地址  //结构体Blockchain的指针类型

}
*/

func shuff(n int) []int { //随机分配噪声节点
	rand.Seed(time.Now().UnixNano())
	var arr []int
	for i := 0; i < n; i++ {
		arr = append(arr, i+1)
	}
	//fmt.Println(arr)
	rand.Shuffle(len(arr), func(i, j int) {
		arr[i], arr[j] = arr[j], arr[i]
	})
	//fmt.Println(arr)
	return arr
}

func initModel() {
	python3.Py_Initialize()
	if !python3.Py_IsInitialized() {
		fmt.Println("Error initializing the python interpreter")
		os.Exit(1)
	}
	path, _ := os.Getwd()
	pyTorchModule = ImportModule(path+"/ML", "client_obj") //引入client_obj.py
	if pyTorchModule == nil {
		log.Fatalf("pyTorchModule is nil")
		os.Exit(1)
	}
	pyInitFunc = pyTorchModule.GetAttrString("init")
	pyPrivFunc = pyTorchModule.GetAttrString("privateFun")
	pyNoise = pyTorchModule.GetAttrString("getNoise")
	pyDownload = pyTorchModule.GetAttrString("updateModel")     //update model by global
	pyUpdate = pyTorchModule.GetAttrString("simpleStep")        //update by grad
	pyAccur = pyTorchModule.GetAttrString("getTestAccur")       //get test accur of model //传入全局模型参数数组并返回训练准确到
	pyCompare = pyTorchModule.GetAttrString("ComparTrainAccur") //compare train set accur before update and after update
	pyKrum = pyTorchModule.GetAttrString("krum")                //krun sorce
	pyComm = pyTorchModule.GetAttrString("comm")                //value of comm  //一维浮点数数组，精度，底数
	pyClose = pyTorchModule.GetAttrString("closeWrite")
	pyInitFunc.CallFunctionObjArgs()
}

func ImportModule(dir, name string) *python3.PyObject { //引入python文件

	sysModule := python3.PyImport_ImportModule("sys")
	path := sysModule.GetAttrString("path")
	//pathStr, _ := pythonRepr(path)
	//log.Println("before add path is " + pathStr)
	python3.PyList_Insert(path, 0, python3.PyUnicode_FromString(""))
	python3.PyList_Insert(path, 0, python3.PyUnicode_FromString(dir))
	//pathStr, _ = pythonRepr(path)
	//log.Println("after add path is " + pathStr)
	return python3.PyImport_ImportModule(name)
}

func pythonRepr(o *python3.PyObject) (string, error) { //PyObject转string
	if o == nil {
		return "", fmt.Errorf("object is nil")
	}
	s := o.Repr()
	if s == nil {
		python3.PyErr_Clear()
		return "", fmt.Errorf("failed to call Repr object method")
	}
	defer s.DecRef()
	return python3.PyUnicode_AsUTF8(s), nil
}

func strToFloatArry(strArry string) ([]float64, error) { //str转float数组
	var strlist, strlft, strright []string
	strlft = strings.Split(strArry, "[")      //去掉[
	strright = strings.Split(strlft[1], "]")  //去掉]
	strlist = strings.Split(strright[0], ",") //去掉,
	//log.Println(strlist)
	var floatArry []float64
	var fo float64
	for i := 0; i < len(strlist); i++ {
		var s []string
		s = strings.Split(strlist[i], " ") //，后面有空格，需要先去除
		for j := 0; j < len(s); j++ {
			if len(s[j]) > 2 { //去掉回车
				fo, _ = strconv.ParseFloat(s[j], 64)
			}
		}
		floatArry = append(floatArry, fo)
	}
	return floatArry, nil
}

func testModel(weights []float64, i int) float64 { //update model by global and calc accur on teat set
	runtime.LockOSThread()

	_gstate := python3.PyGILState_Ensure()

	argArray := python3.PyList_New(len(weights))

	for i := 0; i < len(weights); i++ {
		python3.PyList_SetItem(argArray, i, python3.PyFloat_FromDouble(weights[i]))
	}
	var res float64
	testAccur := pyAccur.CallFunctionObjArgs(argArray, python3.PyLong_FromLong(i))
	str, _ := pythonRepr(testAccur)
	res, _ = strconv.ParseFloat(str, 64)

	python3.PyGILState_Release(_gstate)
	return res

}

func getGrad(globalW []float64) ([]float64, error) { //根据全局模型计算梯度
	runtime.LockOSThread()
	_gstate := python3.PyGILState_Ensure()       //lock()
	argArray := python3.PyList_New(len(globalW)) //转为python类型
	for i := 0; i < len(globalW); i++ {
		python3.PyList_SetItem(argArray, i, python3.PyFloat_FromDouble(globalW[i]))
	}
	var result *python3.PyObject
	result = pyPrivFunc.CallFunctionObjArgs(argArray) //python类型的梯度数组
	if result == nil {
		log.Fatalf("pyPrivFunc is nil")
	}
	strArry, _ := pythonRepr(result) //转换为str
	if strArry == "" {
		log.Fatalf("strArry is nil")
	}

	var floatArrys []float64
	floatArrys, _ = strToFloatArry(strArry) //str->float[]
	python3.PyGILState_Release(_gstate)     //释放锁
	return floatArrys, nil
}

func requestNoise(iterationCount int) ([]float64, error) { //获得噪声数组
	runtime.LockOSThread()
	_gstate := python3.PyGILState_Ensure() //lock()
	var result *python3.PyObject

	result = pyNoise.CallFunctionObjArgs(python3.PyLong_FromLong(iterationCount)) //噪声数组
	if result == nil {
		log.Fatalf("pyNoise is nil")
	}

	strArry, _ := pythonRepr(result) //转换为str
	if strArry == "" {
		log.Fatalf("strArry is nil")
	}

	var floatArrys []float64
	floatArrys, _ = strToFloatArry(strArry) //str->float[]

	python3.PyGILState_Release(_gstate) //释放锁
	return floatArrys, nil

}

func aggredGradLoacal(grad []float64) { //通过梯度更新本地模型
	runtime.LockOSThread()
	_gstate := python3.PyGILState_Ensure() //lock()
	//var argArray *python3.PyObject
	argArray := python3.PyList_New(len(grad)) //转为python类型
	for i := 0; i < len(grad); i++ {
		python3.PyList_SetItem(argArray, i, python3.PyFloat_FromDouble(grad[i]))
	}
	pyUpdate.CallFunctionObjArgs(argArray)

	python3.PyGILState_Release(_gstate) //释放锁

}

func replaceModel(global []float64) { //将本地模型替换为全局模型
	runtime.LockOSThread()
	_gstate := python3.PyGILState_Ensure() //lock()
	argArray := python3.PyList_New(len(global))
	for i := 0; i < len(global); i++ {
		python3.PyList_SetItem(argArray, i, python3.PyFloat_FromDouble(global[i]))
	}
	pyDownload.CallFunctionObjArgs(argArray)

	python3.PyGILState_Release(_gstate) //释放锁

}

func comm(updates []float64) int64 {
	runtime.LockOSThread()
	_gstate := python3.PyGILState_Ensure() //lock()
	argArray := python3.PyList_New(len(updates))
	for i := 0; i < len(updates); i++ {
		python3.PyList_SetItem(argArray, i, python3.PyFloat_FromDouble(updates[i]))
	}
	res := pyComm.CallFunctionObjArgs(argArray)
	if res == nil {
		log.Println("comm err")
	}
	str, _ := pythonRepr(res)
	//log.Println("str is ", str)
	result, _ := strconv.ParseInt(str, 10, 64)

	python3.PyGILState_Release(_gstate)
	return result

}

func testUpdate(global, aggGrad []float64) float64 {

	runtime.LockOSThread()
	_gstate := python3.PyGILState_Ensure() //lock()

	//var argArray *python3.PyObject
	gloArray := python3.PyList_New(len(global))

	for i := 0; i < len(global); i++ {
		python3.PyList_SetItem(gloArray, i, python3.PyFloat_FromDouble(global[i]))
	}

	grdArray := python3.PyList_New(len(aggGrad))
	for i := 0; i < len(global); i++ {
		python3.PyList_SetItem(grdArray, i, python3.PyFloat_FromDouble(aggGrad[i]))
	}
	del := pyCompare.CallFunctionObjArgs(gloArray, grdArray)
	str, _ := pythonRepr(del)
	num, _ := strconv.ParseFloat(str, 64)

	python3.PyGILState_Release(_gstate) //释放锁
	return num

}

func addNoise(grad, noise []float64) []float64 {
	gradInt := make([]int64, len(grad))
	for i := 0; i < len(grad); i++ {
		gradInt[i] = int64(grad[i] * math.Pow10(2))
	}
	for i := 0; i < len(gradInt); i++ {
		grad[i] = float64(gradInt[i] / 100) //防止相加时第三位小数产生进位
	}

	for i := 0; i < len(noise); i++ {
		gradInt[i] = int64(noise[i] * math.Pow10(2))
	}
	for i := 0; i < len(noise); i++ {
		noise[i] = float64(gradInt[i] / 100)
	}

	floatArr := make([]float64, len(gradInt))
	for i := 0; i < len(floatArr); i++ {
		floatArr[i] = grad[i] + noise[i]
	}
	return floatArr

}

func verifyMask(maskde []float64, commgra, commnoi int64) bool {
	res := comm(maskde)
	if res == commgra*commnoi%int64(p) {
		return true
	}
	return false

}

func addGrad(aggred []float64) {
	numofAgg = numofAgg + 1
	for i := 0; i < len(recvedagg); i++ {
		recvedagg[i] = aggred[i] + recvedagg[i]
	}

}

func updateGlobal(global, grads []float64) []float64 {
	for i := 0; i < len(grads); i++ {
		grads[i] = grads[i] / float64(numofAgg) //联邦平均
	}
	numofAgg = 0 //恢复
	for i := 0; i < len(grads); i++ {
		global[i] = global[i] + grads[i]
	}
	for i := 0; i < len(recvedagg); i++ {
		recvedagg[i] = 0
	}
	return global

}

func main() {
	//var client Honest

	//client = Honest{ aggregatedGrad: make([]Update, 0, 5)}

	initModel()
	e = 2 //计算承诺时的精度
	g = 7
	p = 5527

	rand.Seed(time.Now().UnixNano()) //初始的训练开始前的全局模型
	for i := 0; i < 7850; i++ {
		Global = append(Global, rand.Float64())

	}
	for i := 0; i < len(recvedagg); i++ {
		recvedagg[i] = 0
	}

	for i := 0; i < 300; i++ {
		log.Println("--------------------iteration ", i, "---------------------------")
		//itbef := testModel(Global) //
		//log.Println("before update accuracy is", itbef)

		gradf1, _ := getGrad(Global) //计算梯度   python  再跑一次  代表多个训练者
		gradf2, _ := getGrad(Global)
		gradf3, _ := getGrad(Global)
		//gradf4, _ := getGrad(Global)
		gradf5 := make([]float64, 7850)

		rand.Seed(time.Now().UnixNano()) //初始的训练开始前的全局模型
		for j := 0; j < 7850; j++ {
			gradf5 = append(gradf5, 5*100*rand.Float64())

		}

		//var recvGrad [][]float64

		//recvGrad = append(recvGrad, gradf1[:])
		///recvGrad = append(recvGrad, gradf2[:])
		//recvGrad = append(recvGrad, gradf3[:])
		//recvGrad = append(recvGrad, gradf4[:])
		//recvGrad = append(recvGrad, gradf5[:])

		//noused := krumList(recvGrad, 1)
		//log.Println("this is the honest client index", noused)

		addGrad(gradf1)
		addGrad(gradf2)
		addGrad(gradf3)
		//addGrad(gradf4)
		addGrad(gradf5)

		//noiseFormPeer,_ := requestNoise(i)

		//resOfComm := comm(gradf)   //计算承诺供验证者验证
		//resofnoise := comm(noiseFormPeer)

		//noisedGrad := addNoise(gradf,noiseFormPeer)   //把加过噪声的梯度发送给验证者进行投毒攻击检查  //verifier  func   comm  krum
		//flags := verifyMask(noisedGrad,resOfComm,resofnoise)   //判断

		//aggred := addGrad(gradf, gradf) //梯度聚合     //用验证后的结果 再verifier聚合  //恶意 grad 随机生成

		Global = updateGlobal(Global, recvedagg[:])

		itaft := testModel(Global, i)
		log.Println("after update accuracy is", itaft)

	}
	pyClose.CallFunctionObjArgs() //关闭write

	//updated := updateGlobal(Global, gradf)

	//ita := testModel(updated)
	//log.Println("ita is", ita)

	//log.Println(gradf)

	/*

		cc := comm(gradf) //tidu大多数负的，结果会为0  //要全转正
		//log.Println(cc)

		var noiser []float64
		noiser, _ = requestNoise(0)

		cn := comm(noiser)

		var maskgrad []float64

		maskgrad = addNoise(gradf, noiser)

		plea := verifyMask(maskgrad, cc, cn)   //over
		log.Println(plea)
	*/

	//aa := testUpdate(Global, gradf) //over
	//log.Println(aa)

	//log.Println(gradf)

	//aggredGradLoacal(gradf)
	//var sum []float64
	//for i := 0; i < len(Global); i++ {
	//	sum = append(sum, gradf[i]+Global[i])
	//
	//}
	//replaceModel(sum)

	//res := comm(gradf)
	//println(res)

	//acc := testModel(Global) //wanc
	//log.Println(acc)

	/*var noiser []float64
	noiser, _ = requestNoise(0)
	log.Println("start")

	for i := 0; i < len(noiser); i++ {

		log.Println(noiser[i])
	}

	log.Println(len(noiser)) //完成*/ //显示str数组
	/*

		float1 := [6]float64{1.2, 3.3, 1.1, 2.2, 10.5, 1.1}
		float2 := [6]float64{1.1, 3.0, 1.7, 2.5, 9.5, 0.8}
		float3 := [6]float64{1.5, 1.3, 2.1, 2.2, 10.6, 0.7}
		float4 := [6]float64{1.8, 2.3, 1.2, 2.9, 11.5, 2.0}
		float5 := [6]float64{7.2, 9.3, 5.1, 4.2, 3.5, 5.1}

		var num [][]float64
		//num = make([][]float64, 5, 6)   //不能添加
		//num[0] = float1[:]

		//num[0] = append(num[0], float1[:])
		num = append(num, float1[:])
		num = append(num, float2[:])
		num = append(num, float5[:])
		num = append(num, float3[:])
		num = append(num, float4[:])

		//num[0] = float1
		//num[1] = float2
		//num[2] = float3
		//num[3] = float4
		//num[4] = float5
		//for i := 0; i < len(num); i++ {
		//	log.Println(" print", num[i])
		//}

		red := krumList(num, 1)
		//log.Println("list is ", red)
		log.Println(red)*/

}

func krumList(rec [][]float64, clip int64) []int64 { //输入一个人数*7850的二维数组。返回选择的训练者在第一维中的位置i  //2f+2<n
	runtime.LockOSThread()
	_gstate := python3.PyGILState_Ensure() //lock()
	peers := python3.PyList_New(len(rec))
	for i := 0; i < len(rec); i++ {
		grad := python3.PyList_New(len(rec[i]))
		for j := 0; j < len(rec[i]); j++ {
			python3.PyList_SetItem(grad, j, python3.PyFloat_FromDouble(rec[i][j]))
		}
		python3.PyList_SetItem(peers, i, grad)
	}
	indexofCliet := pyKrum.CallFunctionObjArgs(peers, python3.PyLong_FromLong(int(clip)))
	strArry, _ := pythonRepr(indexofCliet) //转换为strlog.println(strArry)
	log.Println("return index", strArry)
	python3.PyGILState_Release(_gstate) //释放锁

	return nil

}

func serectSend(serect []float64) {
	n := 2 * (len(serect) + 1)
	num := make([]float64, n)

	for i := 0; i < n; i++ {
		t := 0.0
		for j := 0; j < n; j++ {
			temp := math.Pow(float64(i), float64(j)) * serect[j]
			t = t + temp
		}
		num[i] = t
	}

}

func addBlockToChain(block Block, id int) {
	// Add the block to chain

	log.Printf(strconv.Itoa(id)+":Adding block for %d, I am at %d\n",
		block.Iteration, iterationCount)
	bc.AddBlock(block)

}
