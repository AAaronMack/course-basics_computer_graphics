const person = {
  firstName: "John",
  lastName : "Doe",
  id       : 5566,
  fullName : function() {
    return this.firstName + " " + this.lastName;
  }
};

console.log(">>>>>>>>>> ", person.id)
console.log(">>>>>>>>>> ", person.firstName)
console.log(">>>>>>>>>> ", person.fullName())
