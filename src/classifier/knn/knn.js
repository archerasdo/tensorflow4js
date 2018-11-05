"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
exports.__esModule = true;
/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
var tf = require("@tensorflow/tfjs");
var tfjs_1 = require("@tensorflow/tfjs");
function concatWithNulls(ndarray1, ndarray2) {
    if (ndarray1 == null && ndarray2 == null) {
        return null;
    }
    if (ndarray1 == null) {
        return ndarray2.clone();
    }
    else if (ndarray2 === null) {
        return ndarray1.clone();
    }
    return ndarray1.concat(ndarray2, 0);
}
function topK(values, k) {
    var valuesAndIndices = [];
    for (var i = 0; i < values.length; i++) {
        valuesAndIndices.push({ value: values[i], index: i });
    }
    valuesAndIndices.sort(function (a, b) {
        return b.value - a.value;
    });
    var kRankValue = valuesAndIndices[k].value;
    // 特征相同 排序需要等距抽样
    if (kRankValue == valuesAndIndices[0].value) {
        var equalDistanceList = valuesAndIndices.filter(function (el) { return el.value == kRankValue; });
        var equalDistanceListLength = equalDistanceList.length;
        var frequency_1 = Math.floor(equalDistanceListLength / k);
        return equalDistanceList
            .filter(function (el, index) { return index % frequency_1 === 0; });
    }
    else {
        return valuesAndIndices.filter(function (el, index) { return index < k; });
    }
}
/**
 * A K-nearest neighbors (KNN) classifier that allows fast
 * custom model training on top of any tensor input. Useful for transfer
 * learning with an embedding from another pretrained model.
 */
var KNNClassifier = /** @class */ (function () {
    function KNNClassifier() {
        // Individual class datasets used when adding examples. These get concatenated
        // into the full trainDatasetMatrix when a prediction is made.
        this.classDatasetMatrices = {};
        this.classExampleCount = {};
    }
    /**
     * Adds the provided example to the specified class.
     */
    KNNClassifier.prototype.addExample = function (example, classIndex) {
        var _this = this;
        if (this.exampleShape == null) {
            this.exampleShape = example.shape;
        }
        if (!tfjs_1.util.arraysEqual(this.exampleShape, example.shape)) {
            throw new Error("Example shape provided, " + example.shape + " does not match " +
                ("previously provided example shapes " + this.exampleShape + "."));
        }
        if (!Number.isInteger(classIndex)) {
            throw new Error("classIndex must be an integer, got " + classIndex + ".");
        }
        this.clearTrainDatasetMatrix();
        tf.tidy(function () {
            var normalizedExample = _this.normalizeVectorToUnitLength(example.flatten());
            var exampleSize = normalizedExample.shape[0];
            if (_this.classDatasetMatrices[classIndex] == null) {
                _this.classDatasetMatrices[classIndex] =
                    normalizedExample.as2D(1, exampleSize);
            }
            else {
                var newTrainLogitsMatrix = _this.classDatasetMatrices[classIndex]
                    .as2D(_this.classExampleCount[classIndex], exampleSize)
                    .concat(normalizedExample.as2D(1, exampleSize), 0);
                _this.classDatasetMatrices[classIndex].dispose();
                _this.classDatasetMatrices[classIndex] = newTrainLogitsMatrix;
            }
            tf.keep(_this.classDatasetMatrices[classIndex]);
            if (_this.classExampleCount[classIndex] == null) {
                _this.classExampleCount[classIndex] = 0;
            }
            _this.classExampleCount[classIndex]++;
        });
    };
    /**
     * This method return distances between the input and all examples in the
     * dataset.
     *
     * @param input The input example.
     * @returns cosine similarities for each entry in the database.
     */
    KNNClassifier.prototype.similarities = function (input) {
        var _this = this;
        return tf.tidy(function () {
            var normalizedExample = _this.normalizeVectorToUnitLength(input.flatten());
            var exampleSize = normalizedExample.shape[0];
            // Lazily create the logits matrix for all training examples if necessary.
            if (_this.trainDatasetMatrix == null) {
                var newTrainLogitsMatrix = null;
                for (var i in _this.classDatasetMatrices) {
                    newTrainLogitsMatrix = concatWithNulls(newTrainLogitsMatrix, _this.classDatasetMatrices[i]);
                }
                _this.trainDatasetMatrix = newTrainLogitsMatrix;
            }
            if (_this.trainDatasetMatrix == null) {
                console.warn('Cannot predict without providing training examples.');
                return null;
            }
            tf.keep(_this.trainDatasetMatrix);
            var numExamples = _this.getNumExamples();
            return _this.trainDatasetMatrix.as2D(numExamples, exampleSize)
                .matMul(normalizedExample.as2D(exampleSize, 1))
                .as1D();
        });
    };
    /**
     * Predicts the class of the provided input using KNN from the previously-
     * added inputs and their classes.
     *
     * @param input The input to predict the class for.
     * @returns A dict of the top class for the input and an array of confidence
     * values for all possible classes.
     */
    KNNClassifier.prototype.predictClass = function (input, k) {
        if (k === void 0) { k = 3; }
        return __awaiter(this, void 0, void 0, function () {
            var knn, kVal, topKItem, _a;
            var _this = this;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        if (k < 1) {
                            throw new Error("Please provide a positive integer k value to predictClass.");
                        }
                        if (this.getNumExamples() === 0) {
                            throw new Error("You have not added any examples to the KNN classifier. " +
                                "Please add examples before calling predictClass.");
                        }
                        knn = tf.tidy(function () { return _this.similarities(input).asType('float32'); });
                        kVal = Math.min(k, this.getNumExamples());
                        _a = topK;
                        return [4 /*yield*/, knn.data()];
                    case 1:
                        topKItem = _a.apply(void 0, [_b.sent(), kVal]);
                        knn.dispose();
                        return [2 /*return*/, this.calculateTopClass(topKItem, kVal)];
                }
            });
        });
    };
    /**
     * Clears the saved examples from the specified class.
     */
    KNNClassifier.prototype.clearClass = function (classIndex) {
        if (this.classDatasetMatrices[classIndex] == null) {
            throw new Error('Cannot clear invalid class ${classIndex}');
        }
        delete this.classDatasetMatrices[classIndex];
        delete this.classExampleCount[classIndex];
        this.clearTrainDatasetMatrix();
    };
    KNNClassifier.prototype.clearAllClasses = function () {
        for (var i in this.classDatasetMatrices) {
            this.clearClass(+i);
        }
    };
    KNNClassifier.prototype.getClassExampleCount = function () {
        return this.classExampleCount;
    };
    KNNClassifier.prototype.getClassifierDataset = function () {
        return this.classDatasetMatrices;
    };
    KNNClassifier.prototype.getNumClasses = function () {
        return Object.keys(this.classExampleCount).length;
    };
    KNNClassifier.prototype.setClassifierDataset = function (classDatasetMatrices) {
        this.clearTrainDatasetMatrix();
        this.classDatasetMatrices = classDatasetMatrices;
        for (var i in classDatasetMatrices) {
            this.classExampleCount[i] = classDatasetMatrices[i].shape[0];
        }
    };
    /**
     * Calculates the top class in knn prediction
     * @param topKItem The closest K items.
     * @param kVal The value of k for the k-nearest neighbors algorithm.
     */
    KNNClassifier.prototype.calculateTopClass = function (topKItem, kVal) {
        var exampleClass = -1;
        var confidences = {};
        if (topKItem.length == 0) {
            // No class predicted
            return { classIndex: exampleClass, confidences: confidences };
        }
        var totalWeight = topKItem.reduce(function (ret, el) { return ret + el.value; }, 0);
        var indicesForClasses = [];
        for (var i in this.classDatasetMatrices) {
            var num = this.classExampleCount[i];
            if (+i > 0) {
                num += indicesForClasses[+i - 1];
            }
            indicesForClasses.push(num);
        }
        var topKCountsForClasses = Array(Object.keys(this.classDatasetMatrices).length).fill(0);
        for (var i = 0; i < topKItem.length; i++) {
            var _a = topKItem[i], topkIndices = _a.index, weight = _a.value;
            for (var classId = 0; classId < indicesForClasses.length; classId++) {
                if (topkIndices < indicesForClasses[classId]) {
                    // 计算权重
                    topKCountsForClasses[classId] = topKCountsForClasses[classId] + weight || 0;
                    break;
                }
            }
        }
        // Compute confidences.
        var topConfidence = 0;
        for (var i in this.classDatasetMatrices) {
            var probability = totalWeight > 0 ? topKCountsForClasses[i] / totalWeight : 0;
            if (probability >= topConfidence) {
                topConfidence = probability;
                exampleClass = +i;
            }
            confidences[i] = probability;
        }
        return { classIndex: exampleClass, confidences: confidences };
    };
    /**
     * Clear the lazily-loaded train logits matrix due to a change in
     * training data.
     */
    KNNClassifier.prototype.clearTrainDatasetMatrix = function () {
        if (this.trainDatasetMatrix != null) {
            this.trainDatasetMatrix.dispose();
            this.trainDatasetMatrix = null;
        }
    };
    /**
     * Normalize the provided vector to unit length.
     */
    KNNClassifier.prototype.normalizeVectorToUnitLength = function (vec) {
        return tf.tidy(function () {
            var sqrtSum = vec.norm();
            return tf.div(vec, sqrtSum);
        });
    };
    KNNClassifier.prototype.getNumExamples = function () {
        var total = 0;
        for (var i in this.classDatasetMatrices) {
            total += this.classExampleCount[+i];
        }
        return total;
    };
    KNNClassifier.prototype.dispose = function () {
        this.clearTrainDatasetMatrix();
        for (var i in this.classDatasetMatrices) {
            this.classDatasetMatrices[i].dispose();
        }
    };
    return KNNClassifier;
}());
exports.KNNClassifier = KNNClassifier;
function create() {
    return new KNNClassifier();
}
exports.create = create;
