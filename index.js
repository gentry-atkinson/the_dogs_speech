let model, mic, result;

function predictBark(scores) {
 // Array of words that the recognizer is trained to recognize.
 const words = ['bark', 'no bark'];
 console.log("Got scores, ", scores);
 //recognizer.listen(({scores}) => {
   // Turn scores into a list of (score,word) pairs.
   //scores = Array.from(scores).map((s, i) => ({score: s, word: words[i]}));
   // Find the most probable word.
   //scores.sort((s1, s2) => s2.score - s1.score);
   //document.querySelector('#console').textContent = scores[0].word;
 //}, {probabilityThreshold: 0.75});


}

async function app() {
  const modelUrl = 'https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1';
  const model = await tf.loadGraphModel(modelUrl, { fromTFHub: true });


  const mic = await tf.data.microphone({
   fftSize: 1024,
   columnTruncateLength: 232,
   numFramesPerSpectrogram: 43,
   sampleRateHz:44100,
   includeSpectrogram: true,
   includeWaveform: true
  });
  const audioData = await mic.capture();
  const spectrogramTensor = audioData.spectrogram;
  spectrogramTensor.print();
  const waveformTensor = audioData.waveform;
  waveformTensor.print();
  mic.stop();
  const waveform = tf.zeros([16000 * 3]);
  const [scores, embeddings, spectrogram] = model.predict(waveform);
  scores.print(verbose=true);  // shape [N, 521]
  embeddings.print(verbose=true);  // shape [N, 1024]
  spectrogram.print(verbose=true);  // shape [M, 64]
  //Find class with the top score when mean-aggregated across frames.
  scores.mean(axis=0).argMax().print(verbose=true);
  //Should print 494 corresponding to 'Silence' in YAMNet Class Map.
  predictBark(scores);
}


app();
