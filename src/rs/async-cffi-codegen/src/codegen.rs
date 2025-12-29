pub trait CodeGen {
    fn codegen(&self, indent: usize) -> String;
}
