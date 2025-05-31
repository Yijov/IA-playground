export function getRandomInt(): number {
  return Math.floor(Math.random() * 1000) + 1;
}


export function shuffleArrays(arr1:number[], arr2:number[]) {
  for (let i = arr1.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr1[i], arr1[j]] = [arr1[j], arr1[i]];
    [arr2[i], arr2[j]] = [arr2[j], arr2[i]];
  }
}