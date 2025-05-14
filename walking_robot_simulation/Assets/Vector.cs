using System;
using UnityEngine;


[Serializable]
public struct SerdeVector2
{
    public float x;
    public float y;

    public static implicit operator Vector2(SerdeVector2 a) => new Vector2(a.x, a.y);
    public static implicit operator SerdeVector2(Vector2 a) => new SerdeVector2{x = a.x, y = a.y};
    public override String ToString() => $"({x}, {y})";

}
[Serializable]
public struct SerdeQuaternion
{
    public float x;
    public float y;
    public float z;
    public float w;


    public static implicit operator Quaternion(SerdeQuaternion a) => new Quaternion{
        x=a.x, y=a.y, z=a.z, w=a.w
    };

    public static implicit operator SerdeQuaternion(Quaternion a) => 
        new SerdeQuaternion{x=a.x,y=a.y,z=a.z, w=a.w};
    public override String ToString() => $"({x}, {y}, {z}, {w})";
}

[Serializable]
public struct SerdeVector3
{
    public float x;
    public float y;
    public float z;


    public static implicit operator Vector3(SerdeVector3 a) => new Vector3(a.x, a.y, a.z);
    public static implicit operator SerdeVector3(Vector3 a) => new SerdeVector3{x = a.x,y =  a.y,z =  a.z};
    public override String ToString() => $"({x}, {y}, {z})";
}

