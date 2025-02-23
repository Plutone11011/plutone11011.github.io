
use std::fmt::format;
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
        self.children = children;
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

fn save_svg_to_file(svg_data: &[u8], file_path: &str) -> Result<()> {
    // Create or truncate the file
    let mut file = File::create(file_path)?;
    // Write the SVG data to the file
    file.write_all(svg_data)?;
    Ok(())
}

fn build_graphviz_op_node(id: &str, label: &str) -> String{
    format!("{}[label=\"{}\"]\n", id, label)
}

fn build_graphviz_data_node(id: &str, label: &str, data: f64, grad: f64) -> String{
    format!("{}[shape={}, label=\"{} | {} | {}\"]\n", id, "square", label, data, grad)
}

fn build_computational_graph(root: &Value, current_op_n: usize) -> String{
    // 1. if op is set,  build node string
    // with specific label for it, as well as shape
    // 2. connect edge from op node to current root node
    // 3. if children are present, create node for both of them
    // 4. create edge connecting both to op node
    // 5. recursion over children

    let mut graphviz_str: String = String::new();
    if root.children.is_empty() {
        return graphviz_str
    }
    let id_op_node = format!("op{}", current_op_n);
    match root.op {
        Op::Mult => {
            graphviz_str.push_str(&build_graphviz_op_node(id_op_node.as_str(), "*"));
        },
        Op::Add => {
            graphviz_str.push_str(&build_graphviz_op_node(id_op_node.as_str(), "+"));
        },
        _ => {}
    }

    graphviz_str.push_str(&format!("{} -> {}\n", id_op_node, root.label));

    for child in root.children.iter(){
        graphviz_str.push_str(&build_graphviz_data_node(&child.label, &child.label, child.data, child.grad));
        graphviz_str.push_str(&format!("{} -> {}\n", child.label , id_op_node));
        graphviz_str.push_str(&build_computational_graph(&child, current_op_n+1));
    }

    graphviz_str

    
}

fn main() {
    let s: &str = "square";
    let mut a = Value::default();
    a.set_data(2.0);
    a.set_label("a");
    let mut b = Value::default();
    b.set_data(-3.0);
    b.set_label("b");

    let mut c = a+b;
    c.set_label("c");
    let mut d = Value::default();
    d.set_data(1.0);
    d.set_label("d");

    let mut L = c*d;
    L.set_label("L");

    

    let graph_str = format!(r#" strict digraph Comp {{
                {}
            "#, build_graphviz_data_node(L.label.as_str(), L.label.as_str(), L.data, L.grad));
    
    let final_str = format!("{} {}}}", graph_str, build_computational_graph(&L, 0));
    println!("{}", final_str);
    // let g: Graph = parse(
    //     &final_str).unwrap();
    let g = parse(&final_str).unwrap();

    // L[shape=square]
    // L[label="L 4.0"]
    // op1 -> L
    // op1[label = "+"]
    // d[label = "d -6.0"]
    // c[label = "c 10.0"]
    // c[shape=square]
    // d[shape=square]
    // c -> op1
    // d -> op1

    let graph_svg = exec(
        g,
        &mut PrinterContext::default(),
        vec![Format::Svg.into()],
    )
    .unwrap();

    let _ = save_svg_to_file(&graph_svg, "comp_graph.svg");
} 
