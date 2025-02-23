
use std::ops::{Add, Mul, Sub, Div};
use std::fs::File;
use std::io::*;
use graphviz_rust::dot_generator::*;
use graphviz_rust::dot_structures::*;
use graphviz_rust::{
    attributes::{GraphAttributes, NodeAttributes},
    cmd::{CommandArg, Format},
    exec, exec_dot, parse,
    printer::{DotPrinter, PrinterContext},
};


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
    backward: Option<Box<dyn FnMut()>>,
    label: String 
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

    fn set_label(&mut self, label: &str) -> (){
        self.label = label.to_string();
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
        out.set_label("+");
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
        out.set_label("*");
        out.set_children(vec![self, rhs]);
        
        out
    }
}

fn save_svg_to_file(svg_data: &[u8], file_path: &str) -> Result<()> {
    // Create or truncate the file
    let mut file = File::create(file_path)?;
    // Write the SVG data to the file
    file.write_all(svg_data)?;
    Ok(())
}

fn main() {
    let g: Graph = parse(
        r#"
        strict digraph Comp {
            
            L[shape=square]
            L[label="L 4.0"]
            op1 -> L
            op1[label = "+"]
            d[label = "d -6.0"]
            c[label = "c 10.0"]
            c[shape=square]
            d[shape=square]

            c -> op1
            d -> op1
            
        }
        "#,
    ).unwrap();


    let graph_svg = exec(
        g,
        &mut PrinterContext::default(),
        vec![Format::Svg.into()],
    )
    .unwrap();

    let _ = save_svg_to_file(&graph_svg, "comp_graph.svg");
} 
