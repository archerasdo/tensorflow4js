"use strict";

const tf = require('@tensorflow/tfjs')
const math = require('mathjs')
const knnClassifier = require('./knn/knn')
math.config({
  number: 'BigNumber'
})

class KnnClassifier {
  /**
   * dataSet的数据结构
   * {
   *    特征向量
   *    feature: Array<Number>,
   *    类别标签
   *    label: String
   * }
  */
  
  /**
   * knn 分类
   * @param {Array<dataSet>} trainSet 数据集
   * @param {Array<String>} label 分类类别标签
  */
  constructor(trainSet, label) {
    this.trainSet = trainSet
    this.label = label
    this.tensorModel = knnClassifier.create()
  }

  train() {
    const train = this.trainSet.reduce((ret, el) => {
      const { label: type = '', feature = [] } = el
      const label = this.label[type]

      if (ret[label]) {
        return {
          ...ret,
          [label]: [
            ...ret[label],
            feature
          ]
        }
      } else {
        return {
          ...ret,
          [label]: [ feature ]
        }
      }
    }, {})

    const tfTrain = Object.keys(train).reduce((ret, label) => {
      const tensor = tf.tensor2d(train[label])
      
      return {
        ...ret,
        [+label]: tensor
      }
    }, {})
    
    this.tensorModel.setClassifierDataset(tfTrain)
  }
  
 
  get trainSet() {
    return this._train
  }

  /**
  * @param {Array<dataSet>} trainSet 数据集
  */
  set trainSet(trainSet) {
   if (!Array.isArray(trainSet)) {
     throw new Error('训练数据集必须是一个数组！')
   }
    this._train = trainSet
  }

  get label() {
    return this._label 
  }

  /**
  * @param {Array<String>} label 分类类别标签
  */
  set label(label) {
   if (!Array.isArray(label)) {
     throw new Error('分类标签必须是一个数组！')
   }
    this._label = this._getEnumMap(label)
  }

  /**
   * 分类枚举
   * @param {Array<String>} label 
   */
  _getEnumMap (label) {
    return label.reduce((ret, el, index) => {
      if (!ret[el]) {
        return {
          ...ret,
          [el]: index
        }
      } else {
        return ret
      }
    }, {})
  }

  /**
   * knn 分类预测
   * @param {dataSet} predict 待预测数据
   * @param {Number} k [0-30] knn中的k值
   */
  async predict(predict, k = Math.floor(Math.sqrt(this.trainSet.length))) {
    const { feature = [] } = predict
    const predictTensor = tf.tensor1d(feature)
    const labelMap = this.label

    if (Number.isNaN(k)) {
      throw new Error('k必须是一个数值！')  
    }
    const { classIndex, confidences } = await this.tensorModel.predictClass(predictTensor, k)
    const translated = {
      classIndex: Object.keys(labelMap)[+classIndex],
      confidences: Object.keys(confidences).reduce((ret, el) => {
        const name = Object.keys(labelMap)[+el]

        return {
          ...ret,
          [name]: confidences[el]
        }
      }, {})
    }

    return translated
  }


  /**
   * knn 测试
   * @param {Number} k [0-30] knn中的k值
   */
  async test(k = Math.floor(Math.sqrt(this.trainSet.length))) {
    const { train, test: testWithResult } = this._getSet('bucket')
    const label = Object.keys(this.label)
    this.trainSet = train
    this.train()

    //初始化混淆矩阵
    let matrix = []
    for (const item of label) {
      matrix.push(new Array(label.length).fill(0))
    }

    // 计算混淆矩阵
    for (const test of testWithResult) {
      const tensor = tf.tensor1d(test.feature)
      const { classIndex } = await this.tensorModel.predictClass(tensor, k)
      const testLabel = this.label[test.label]
      const lastMatrixValue = matrix[classIndex][testLabel]


      matrix[classIndex][testLabel] = lastMatrixValue + 1
    }

    let setNum = testWithResult.length
    let eyeCount = 0
    let rowSum = []
    let colSum = []
    let positiveTrueNum = 0
    matrix.forEach((item, index) => {
      const total = item.reduce((ret, el) => ret + el, 0)
      const positiveTrue = item[index]
      positiveTrueNum = positiveTrueNum + positiveTrue
      eyeCount = math.add(eyeCount, positiveTrue)
      let sum = 0
      matrix.forEach((inner, innerIndex) => {
        sum = sum + inner[index]
      })
      colSum.push(sum)
      rowSum.push(total)
      console.log(`class ${index} accuracy: ${total ? math.divide(positiveTrue, total) : 0}`)
    })
    // 计算kappa
    const p0 = math.divide(eyeCount, setNum)
    const pc = math.divide(rowSum.reduce((ret, el, i) => {
      return ret + math.chain(el).multiply(colSum[i]).done()
    },0), math.pow(setNum, 2))
    const kappa =  math.divide(math.add(p0, -pc), math.add(1, -pc))
    console.log(`total accuracy: ${setNum ? math.divide(positiveTrueNum, setNum) : 0}`)
    console.log(`kappa: ${math.format(kappa, {precision: 14})}`)
    console.log(`confusion matrix: `, matrix)    
  }
  
  /**
   * 拆分训练集和测试集
   * @param {BigDecimal} 拆分比例(测试数据集/总数据集)  rate
   * @param {string} 抽样方式  mode ['average', 'random' ]
   */
  _getSet(mode = 'random', rate = 1) {
    const dataSet = this.trainSet
    const trainMap = dataSet.reduce((ret, el) => {
      const { label: type = '', feature = [] } = el
      const label = this.label[type]

      if (ret[label]) {
        return {
          ...ret,
          [label]: [
            ...ret[label],
            el
          ]
        }
      } else {
        return {
          ...ret,
          [label]: [ el ]
        }
      }
    }, {})

    switch (mode) {
      // 随机抽样
      case 'random':
        const testLength = math.multiply(dataSet.length, rate)
        let train = [ ...dataSet ]
        let test = []
  
        for (let i = 0; i < testLength; i ++) {
          const random = Math.floor(math.multiply(math.random(), dataSet.length - i))
          
          train = train.filter((el, index) => index !== random)
          test.push(dataSet[random])
        }
      
        return {
          train,
          test
        }
      // 等距抽样
      case 'average':
        const freq = math.divide(1, rate)
        
        return dataSet.reduce((ret, el, i) => {
          if ((i + 1) % freq) {
            return {
              ...ret,
              train: [ ...ret.train, el ]
            }
          } else {
            return {
              ...ret,
              test: [ ...ret.test, el ]
            }
          }
        }, {
          train: [],
          test: []
        })
      // 十折验证法
      case 'bucket':
        const bucketNum = 10
        let bucketList = []
        for (let i = 0; i < bucketNum; i ++) {
          bucketList.push([])
        }
        const randomIndex = Math.floor(math.random() * 10)

        Object.keys(trainMap).forEach(key => {
          const trainSet = trainMap[key]
          let currentBucket = 0
          
          for (const item of trainSet) {
            bucketList[currentBucket].push(item)
            currentBucket = (currentBucket + 1) % bucketNum
          }
        })
        

        return {
          train: bucketList.reduce((ret, el, i) => i !== randomIndex ? [ ...ret, ...el] : ret , []),
          test: bucketList[randomIndex],
          bucketList
        }

      default: {
        return {
          train: dataSet,
          test: []
        }
      }
    }
  }
}  


module.exports = KnnClassifier