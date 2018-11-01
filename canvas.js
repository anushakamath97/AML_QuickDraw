async function run(decoder_input)
{
  console.log("Run Function");
  //load the model
  model = await tf.loadModel('http://localhost/AML_Project/model/model.json');
  console.log("Model loaded");
  sample_input = tf.randomNormal(decoder_input.shape);
  const pred = model.predict([decoder_input,sample_input]).dataSync();

  console.log("Prediction Done");
   op = tf.round(tf.tensor3d(pred,sample_input.shape));
  div = document.createElement('div');
  div.innerHTML = op;
  document.body.appendChild(div);
}
