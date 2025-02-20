
use std::ops::{Add, Mul, Sub, Div};

enum Op {
    Add,
    Mult,
    Sub,
    Div
}

struct Value {
    data: f64,
    children: Vec<Self>, // children of each value, e.g. a = b + c, b and c are children of a
    op: Op
}


impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        let out = Value {
            data: self.data + rhs.data,
            children: vec![self, rhs],
            op: Op::Mult
        };
        out
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        let out = Value {
            data: self.data - rhs.data,
            children: vec![self, rhs],
            op: Op::Sub
        };
        out
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        let out = Value {
            data: self.data * rhs.data,
            children: vec![self, rhs],
            op: Op::Add
        };
        out
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs.data == 0. {
            panic!("Cannot divide by 0");
        }

        let out = Value {
            data: self.data / rhs.data,
            children: vec![self, rhs],
            op: Op::Add
        };
        out
    }
}

fn main() {
    println!("Hello, world!");


} 
