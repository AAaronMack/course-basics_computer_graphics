// ----> 1 Function parameter type declaration & Downleveling
function greet(person: string, date: Date) {
  console.log(`Hello ${person}, today is ${date.toDateString()}!`);
}

// greet("Maddison", Date());  // error: Argument of type 'string' is not assignable to parameter of type 'Date'.
greet("Maddison", new Date());

"use strict"; // like preventing you from using undeclared variables.
function greet1(person, date) {
  console.log("Hello " + person + ", today is " + date.toDateString() + "!");
}
greet1("Maddison", new Date());