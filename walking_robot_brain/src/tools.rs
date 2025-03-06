pub trait UsedInTrait: Sized{
 	fn used_in<O, F: FnOnce(Self) -> O>(self, f: F) -> O{
		f(self)
	}
}
impl<T: Sized> UsedInTrait for T{}