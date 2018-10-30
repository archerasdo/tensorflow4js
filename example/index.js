"use strict";

const { knn } = require('../src/index')
const fs = require('fs')
const path = require('path')
const co = require('co')


const dataPath = path.resolve('./data.csv')
function getFeatures(featureTable, features) {
  return featureTable.map(f => {
    if (features.find(el => el === f)) {
      return 1
    } else {
      return 0
    }
  })
}



try {
  const data = fs.readFileSync(dataPath)
  const lines = data.toString().split(/\r|\n/)
  const train = []
  const features = []
  const label = []

  for (const items of lines) {
    if (!items) continue
    const itemList = items.split(',')
    let f = itemList[0].replace(/\"/g, '').split('.')
    let type = itemList[1]

    for (let word of f) {
      if (!features.includes(word)) {
        features.push(word)
      }
    }
    if (!label.includes(type)) {
      label.push(type)
    }
  }

  for (const items of lines) {
    if (!items) continue
    const itemList = items.split(',')
    let f = itemList[0].replace(/\"/g, '').split('.')
    let type = itemList[1]

    train.push({
      feature: getFeatures(features, f),
      label: type
    })
  }

  const classifier = new knn(train, label)
  console.log(label)
  classifier.train()
  // classifier.predict({ feature: getFeatures(features, ['status', 'map']) }, 5).then(res => {
  //   console.log(res)
  // })

  classifier.test(5).then(res => {
    console.log(res)
  })

} catch (e) {
  console.log(e.message || e)
}

