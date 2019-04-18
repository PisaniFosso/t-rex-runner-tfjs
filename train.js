const tf = require("@tensorflow/tfjs");
var nj = require('numj');
const trainTestSplit = require('train-test-split');
global.fetch = require('node-fetch');
// require("@tensorflow/tfjs-node");z
require("tfjs-node-save");

require('fs');
const { Image, createCanvas } = require('canvas');

const canvas = createCanvas(100, 56);
const ctx = canvas.getContext('2d');

const Jump = 'Jump-aug';
const Bend = 'Accroupi-aug';
const Negative = 'Other-aug';
const numClasses = 3;

//Shuffle the dataset 
async function shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

async function train_data(X_train, Y_train, X_val, Y_val, model){
    console.log('......History.......');
    for(let i=0;i<50;i++){
     let res = await model.fit(X_train, Y_train, {epochs: 1, validationData:(X_val, Y_val)} );
     console.log('Iteration : %s, loss: %s, accuracy: %s', i, res.history.loss[0].toFixed(5), res.history.acc[0].toFixed(5));
  }
}

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function Mobilenet(input, mobilenet) {
    const Layer = 'global_average_pooling2d_1';
    if(input==null){
        return mobilenet;
    }

    // Return a new output for our custum model.
    const output = tf.tensor1d(mobilenet.infer(input, Layer));
    return output;
}

//Collect data from path
const main = async () => {

const jumps = require('fs')
    .readdirSync(Jump)
    .filter(f => f.endsWith('.jpg'));

const bends = require('fs')
.readdirSync(Bend)
.filter(f => f.endsWith('.jpg'));

const negatives = require('fs')
.readdirSync(Negative)
.filter(f => f.endsWith('.jpg'));


const dataset = [];

//Store data in array
for(let i =0; i < negatives.length; i++){
    dataset.push([negatives[i], [0,0,1]])
}

for(let i=0; i < bends.length; i++){
    dataset.push([bends[i], [0,1,0]])
}

for(let i=0; i< jumps.length; i++){
    dataset.push([jumps[i], [1,0,0]])
}


console.log(dataset.length, 'datas');
shuffle(dataset); //shuflle the data
//train test validation split
const [train, x_test] = trainTestSplit(dataset, 0.9, 1234);
const [x_train, x_val] = trainTestSplit(train, 0.85, 1234);


console.log(x_train.length, 'training images');
console.log(x_test.length, 'test images');
console.log(x_val.length, 'validation');



console.log('Building the training set');


let X_train = tf.tensor([]);
let Y_train = tf.tensor([]);

let X_test = tf.tensor([]);
let Y_test = tf.tensor([]);

let X_val = tf.tensor([]);
let Y_val = tf.tensor([]);


//build the training dataset
console.log('loading Train data...');
for(let i=0; i < x_train.length; i++){
   // From a local file path:
    const img = new Image()
    img.onload = () => ctx.drawImage(img, 0, 0);
    img.onerror = err => { throw err };
    img.src = './' + x_train[i][0].split("_")[0] + '-aug' + '/'+ x_train[i][0];
    var data = [];
    for(let x =0; x < img.width; x++)
    {
        for(let y =0; y < img.height; y++)
        {
            var pixel = ctx.getImageData(x, y, 1, 1);
            ctx.putImageData(pixel, 0, 0);
        }
    }

    X_train = tf.concat([X_train, tf.browser.fromPixels(canvas)]);
    Y_train = tf.concat([Y_train, tf.tensor(x_train[i][1])]);
}

//build the test dataset
console.log('loading Test...');
for(let i=0; i < x_test.length; i++){
    // From a local file path:
     const img = new Image()
     img.onload = () => ctx.drawImage(img, 0, 0);
     img.onerror = err => { throw err };
     img.src = './' + x_test[i][0].split("_")[0] + '-aug' + '/'+ x_test[i][0];

     var data = [];
     for(let x =0; x < img.width; x++)
     {
         for(let y=0; y < img.height; y++)
         {
             var pixel = ctx.getImageData(x, y, 1, 1);
             ctx.putImageData(pixel, 0, 0);
         }
     }
     X_test = tf.concat([X_test, tf.browser.fromPixels(canvas)]);
     Y_test = tf.concat([Y_test, tf.tensor(x_test[i][1])]);
 }

 //build the validation data set
console.log('loading validation data...');
 for(let i=0; i < x_val.length; i++){
    // From a local file path:
     const img = new Image()
     img.onload = () => ctx.drawImage(img, 0, 0);
     img.onerror = err => { throw err };
     img.src = './' + x_val[i][0].split("_")[0] + '-aug' + '/'+ x_val[i][0];
     var data = [];
     for(let x =0; x < img.width; x++)
     {
         for(let y =0; y < img.height; y++)
         {
             var pixel = ctx.getImageData(x, y, 1, 1);
             ctx.putImageData(pixel, 0, 0);
         }
     }
     X_val = tf.concat([X_val, tf.browser.fromPixels(canvas)]);
     Y_val = tf.concat([Y_val, tf.tensor(x_val[i][1])]);
 }

console.log('reshape the data')
X_val = X_val.reshape([-1, 100, 56, 3]);
X_test = X_test.reshape([-1, 100, 56, 3]);
X_train = X_train.reshape([-1, 100, 56, 3]);
Y_train = Y_train.reshape([-1, 3]);
Y_test = Y_test.reshape([-1, 3]);
Y_val = Y_val.reshape([-1, 3]);


console.log(X_val.shape[0], ' validation images');
console.log(X_test.shape[0], ' test images');
console.log(X_train.shape[0], ' training images');

//Import MobileNetV1
console.log('Loading mobileNetV1...')
const mobilenet = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

//adapt the data to fit the model
console.log("loading training images...");
for (let index = 0; index < X_train.shape[0]; index++) {
    X_train[index] = await Mobilenet(X_train[index], mobilenet)
}


console.log("loading test images...");
for (let index = 0; index < X_test.shape[0]; index++) {
    X_test[index] = await Mobilenet(X_test[index], mobilenet)
}

console.log("loading validation images...");
for (let index = 0; index < X_val.shape[0]; index++) {
    X_val[index] = await Mobilenet(X_val[index], mobilenet)
}

mobilenet.summary();

// rather than adding layers to the mobilenet model, we "freeze" the weights
// of the mobilenet model, and only train weights from the new model.

const InputShape = [100, 56, 3]
const flatten = tf.layers.flatten();

//Create the model
const model = tf.sequential();
model.add(tf.layers.inputLayer({ inputShape: InputShape }));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: numClasses, activation: 'softmax', kernelInitializer: 'VarianceScaling', }));


console.log("custom model");
        
        // We use categoricalCrossentropy which is the loss function we use for
        // categorical classification which measures the error between our predicted
        // probability distribution over classes (probability that an input is of each
        // class), versus the label (100% probability in the true class)>
        await model.compile({
            optimizer: tf.train.adam(1e-6),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });

        model.summary();

        //train the model on training dataset
    await train_data(X_train, Y_train, X_val, Y_val , model);
    console.log('========== Saving the model ===========');
    try {
        let tsModelTraining = await model.save('file://punch_kick_simplified');
        console.log('============= model succesfully saved ==============');
      } 
      catch (error) {
        // Handle the error in here
        console.log('============= model error saved ==============');
        console.log(error);
      }
    console.log('************* Model Prediction ****************');
    const y_pred = model.predict(X_test, {batchSize: 4});

    //compute the metrics
    const axis = 1;
    const y_true = Y_test.argMax(axis);
    const y_predicted = y_pred.argMax(axis);

    console.log('************* Model confusionMatrix ****************');
    const cm = tf.math.confusionMatrix(y_true, y_predicted, numClasses);
    cm.print();

    console.log('************* Model binary Accuracy ****************');
    const binaryAccuracy = tf.metrics.binaryAccuracy(y_true, y_predicted);
    binaryAccuracy.print();
    
    console.log('************* Model precision ****************');
    const precision = tf.metrics.precision(y_true, y_predicted);
    precision.print();

    console.log('************* Model recall ****************');
    const recall = tf.metrics.recall(y_true, y_predicted);
    recall.print();

};


main();

