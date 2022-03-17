class Animal {
  constructor(color) {
    this.color = color;
  }

  shout(){
    console.log("XXX~");
  }
}

class Cat extends Animal {
    constructor(color) {
        super(color);
    }
    shout() {
        console.log("Miao~");
    }
}

class Dog extends Animal {
    constructor(color) {
        super(color);
    }
    shout() {
        console.log("Wang~");
    }
}
class Bird extends Animal {
      constructor(color) {
        super(color);
    }
}

let cat = new Cat("Orange");
let dog = new Dog("Yellow");
let bird = new Bird("white");
cat.shout();
dog.shout();
bird.shout();