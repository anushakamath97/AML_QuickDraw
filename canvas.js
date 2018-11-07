async function run(decoder_input)
{
  console.log("Run Function");
  //load the model
  model = await tf.loadModel('http://localhost/AML_Project/model/model.json');
  console.log("Model loaded");
  sample_input = tf.randomNormal(decoder_input.shape);
  const pred = model.predict([decoder_input,sample_input]).dataSync();

  console.log("Prediction Done");
  //op = tf.reshape(tf.round(pred),sample_input.shape);

  result = []
  ind = -1;

  for (var i = 0; i < pred.length; i++) {
    if(i%5 == 0)
    { result.push([]);
      ind += 1;
    }
    result[ind].push(Math.round(pred[i]));
  }

  canvas = document.getElementById("can");
  ctx = canvas.getContext("2d");
  prevX = 0
  prevY = 0

  for (var i = 0; i < result.length; i++) {
    currX = prevX + result[i][0];
    currY = prevY + result[i][1];
    ctx.beginPath();
    ctx.fillRect(currX, currY, 2, 2);
    ctx.closePath();
    prevX = currX;
    prevY = currY;
  }
}
