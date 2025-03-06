use json::{object::Object, JsonValue};
use nalgebra::{Vector2, Vector3};

pub trait TryFromJson: Sized{
	fn try_from_json(json: &JsonValue) -> Result<Self, anyhow::Error> ;
}

pub trait JsonExts: Sized{
	fn as_object(&self) -> Option<&Object>;
	fn as_vector2(&self) -> Option<Vector2<f32>>;
	fn as_vector3(&self) -> Option<Vector3<f32>>;
	fn try_as<T: TryFromJson>(&self) -> Option<T>;
}


impl JsonExts for JsonValue{
	fn as_object(&self) -> Option<&Object> {
		match self{
			JsonValue::Object(object) => Some(object),
			_ => None	
		}
	}
	fn as_vector2(&self) -> Option<Vector2<f32>>{
		let self_obj = self.as_object()?;
		let x = self_obj.get("x").and_then(|x| x.as_f32())?;
		let y = self_obj.get("y").and_then(|x| x.as_f32())?;
		Some(Vector2::new(x, y))
	}
	fn as_vector3(&self) -> Option<Vector3<f32>>{
		let self_obj = self.as_object()?;
		let x = self_obj.get("x").and_then(|x| x.as_f32())?;
		let y = self_obj.get("y").and_then(|x| x.as_f32())?;
		let z = self_obj.get("z").and_then(|x| x.as_f32())?;
	
		Some(Vector3::new(x, y, z))
	}
	
	fn try_as<T: TryFromJson>(&self) -> Option<T>{
		T::try_from_json(self).ok()
	}
}
pub trait ToJson{
	fn to_json(&self) -> JsonValue;
}