
use std::ops::{Add, Mul, Sub, Div};

#[derive(Default)]
enum Op {
    Add,
    Mult,
    Sub,
    Div,
    // NoOp for leaf (input) nodes that are not composed from other functions
    #[default]
    NoOp
}

#[derive(Default)]
struct Value {
    data: f64,
    children: Vec<Self>, // children of each value, e.g. a = b + c, b and c are children of a
    op: Op,
    grad: f64,
    backward: Option<Box<dyn FnMut()>>
} 


impl Value {
    // fn new(data: f64, children: Vec<Value>, op: Op) -> Self {

    //     Value { data: data, children: children, op: op, grad: 0., backward: None }
    
    // }

    fn set_children(&mut self, children: Vec<Value>) -> (){
        self.children = children
    }

    fn set_op(&mut self, op: Op) -> (){
        self.op = op
    }
    
    fn set_gradient(&mut self, grad: f64) -> (){
        self.grad = grad
    }

    fn set_data(&mut self, data: f64) -> (){
        self.data = data
    }

    fn set_backward(&mut self, backward: Option<Box<dyn FnMut()>>) -> () {
        self.backward = backward;
    }
}

impl Add for Value {
    type Output = Value;


    fn add(mut self, mut rhs: Self) -> Self::Output {

        let mut out = Value::default();


        let backward = move || {
            self.grad = 1. * out.grad;
            rhs.grad = 1. * out.grad;
        };
        out.set_backward(Some(Box::new(backward)));
        out.set_data(self.data + rhs.data);
        out.set_op(Op::Add);
        out.set_children(vec![self, rhs]);
        
        out
    }
}


impl Mul for Value {
    type Output = Value;

    fn mul(mut self, mut rhs: Self) -> Self::Output {
        let mut out = Value::default();


        let backward = move || {
            self.grad = rhs.data * out.grad;
            rhs.grad = self.data * out.grad;
        };
        out.set_backward(Some(Box::new(backward)));
        out.set_data(self.data * rhs.data);
        out.set_op(Op::Mult);
        out.set_children(vec![self, rhs]);
        
        out
    }
}



fn main() {
    println!("Hello, world!");


} 
