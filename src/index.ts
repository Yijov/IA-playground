import { getRandomInt } from "./helpers.js";
import { NeuralNetwork } from "./NewralNetwork.js";


let number: number;
const trainButton = document.getElementById("train");
const feedButton = document.getElementById("feed");
const numberDisplay = document.getElementById("number");
const updateButton = document.getElementById("update");
const evalButton = document.getElementById("eval");
const saveButton = document.getElementById("save");
const loadButton = document.getElementById("load");
updateNumber();
const nn = new NeuralNetwork([4, 10, 10, 2]); // 2 inputs, 4 hidden neurons, 1 output


trainButton?.addEventListener("click", () => {
   const inputs: number[][] = [];
  const targets: number[][] = [];
  const sampleCount = 800;

  for (let i = 0; i < sampleCount; i++) {
    const num = getRandomInt();
    inputs.push(Normalize(num));  // input shape: array of arrays, each with 1 normalized number
    targets.push(num % 2 === 0 ? [0, 1] : [1, 0]);
  }

  const epochs = 1000; // or tune as needed
  const learningRate = 0.05;

  nn.train(inputs, targets, epochs, learningRate);

  console.log("Training complete");
});

evalButton?.addEventListener("click", () => {
    const testSamples = 200;
  let correctCount = 0;

  for (let i = 0; i < testSamples; i++) {
    const num = getRandomInt();
    const input = Normalize(num);
    const output = nn.predict(input);  // returns number[]

    // Assuming output has two values: [oddScore, evenScore]
    const predictedClass = output[0] > output[1] ? "odd" : "even";
    const actualClass = num % 2 === 0 ? "even" : "odd";

    if (predictedClass === actualClass) correctCount++;

    console.log(`Input: ${num} -> Output: ${output.map(x => x.toFixed(3))} -> Predicted: ${predictedClass}, Actual: ${actualClass}`);
  }

  const accuracy = (correctCount / testSamples) * 100;
  console.log(`Accuracy: ${accuracy.toFixed(2)}%`);

});

feedButton?.addEventListener("click", () => {
 const input= Normalize(number)
  const output = nn.predict(input);
  console.log(`Input: ${number} ->${input} -> Output: ${output.toString()}`);
});

updateButton?.addEventListener("click", () => {
  updateNumber();
});

saveButton?.addEventListener("click", () => {
  nn.save();
});

loadButton?.addEventListener("click", () => {
 const brain= NeuralNetwork.load();;
 if(!brain) return null
 nn.biases=brain?.biases
 nn.layers=brain?.layers
 nn.weights=brain?.weights
});


function updateNumber() {
  number = getRandomInt();
  if (numberDisplay) numberDisplay.innerText = number.toString();
}


function Normalize(n: number): number[] {
  if (n < 1 || n > 1000) {
    throw new Error("Number must be between 1 and 1000");
  }

  const str = n.toString().padStart(4, '0');
  return str.split('').map(n=>Number(n)/10);
}
