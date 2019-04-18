import * as tf from '@tensorflow/tfjs';
import * as mobileNet from '@tensorflow-models/mobilenet';

navigator.mediaDevices
  .getUserMedia({
    video: true,
    audio: false
  })
  .then(stream => {
    video.srcObject = stream;
  });

  function Keypress(key){
    var event = document.createEvent("Event");
    event.keyCode = key;
    event.key = key;
    event.initEvent('keydown');
    document.dispatchEvent(event);
    event.initEvent('keyup');
    // document.dispatchEvent(event);
    setTimeout(function(){document.dispatchEvent(event);},10)
    

  }
// let KEYUP_DOWN = new KeyboardEvent("keydown", {
//     keyCode: 38
// });

// let  KEYUP_UP = new KeyboardEvent("keyup", {
//   keyCode: 38
// });

// let KEYDOWN_DOWN = new KeyboardEvent("keydown", {
//   keyCode: 40
// });

// let KEYDOWN_UP = new KeyboardEvent("keyup", {
//   keyCode: 40
// });



const video = document.getElementById('cam') as HTMLVideoElement;
const Layer = 'global_average_pooling2d_1';
const mobilenetInfer = m => (p): tf.Tensor<tf.Rank> => m.infer(p, Layer);
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const crop = document.getElementById('crop') as HTMLCanvasElement;

const ImageSize = {
  Width: 100,
  Height: 56
};

const grayscale = (canvas: HTMLCanvasElement) => {
  const imageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
    data[i] = avg;
    data[i + 1] = avg;
    data[i + 2] = avg;
  }
  canvas.getContext('2d').putImageData(imageData, 0, 0);
};

let mobilenet: (p: any) => tf.Tensor<tf.Rank>;
tf.loadModel('http://localhost:5000/model.json').then(model => {
  mobileNet
    .load()
    .then((mn: any) => {
      mobilenet = mobilenetInfer(mn);
      document.getElementById('playground').style.display = 'table';
      document.getElementById('loading-page').style.display = 'none';
      console.log('MobileNet created');
    })
    .then(() => {
      setInterval(() => {
        canvas.getContext('2d').drawImage(video, 0, 0);
        crop.getContext('2d').drawImage(canvas, 0, 0, ImageSize.Width, ImageSize.Height);

        crop
          .getContext('2d')
          .drawImage(
            canvas,
            0,
            0,
            canvas.width,
            canvas.width / (ImageSize.Width / ImageSize.Height),
            0,
            0,
            ImageSize.Width,
            ImageSize.Height
          );

          
          // let img = tf.fromPixels(crop).reshape([-1, 1024]);
          // img = tf.reshape(img, [1, 100, 56, 3]);
        // crop = tf.reshape(crop, [100, 56, 3])
        grayscale(crop);
        const [jump, bend, nothing] = Array.from((model.predict(
          mobilenet(tf.fromPixels(crop))
        ) as tf.Tensor1D).dataSync() as Float32Array);
        const detect = (window as any).Detect;
        if (nothing >= 0.4) {
          return;
        }
        console.log(jump.toFixed(2), bend.toFixed(2));
        if (bend > jump && bend >= 0.25) {
          console.log('%cbend: ' + bend.toFixed(2), 'color: red; font-size: 30px');
          Keypress(40);

          return;
        }
        if (jump > bend && jump >= 0.25) {
          console.log('%cjump: ' + jump.toFixed(2), 'color: blue; font-size: 30px');
          Keypress(32);

          // window.dispatchEvent(KEYDOWN_DOWN);
          // window.dispatchEvent(KEYDOWN_UP);
          return;
        }
      }, 100);
    });
});

