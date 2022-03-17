var i = 5;
while (i < 10) {
  i++;
  console.log("The number is " + i);
} //Output: 6,7,8,9,10

console.log("--------------------------------")

i = 5;
while (i < 10) {
  i++;
  console.log("The number is " + i);

  if(i == 8){
    break;
  }
} //Output: 6,7,8

console.log("--------------------------------")

i = 5;
while (i < 10) {
  i++;
  if(i === 7){
    continue;
  }
  console.log("The number is " + i);

  if(i == 8){
    break;
  }
}//Outputs: 6,8