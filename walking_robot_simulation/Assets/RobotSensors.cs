using System;
using UnityEngine;

public class RobotHeadSensors : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    ArticulationBody body;
    public Target Target;
    public Transform Orientation;

    void Start()
    {
        this.body = this.GetComponent<ArticulationBody>();
    }

    public Vector3 GetSpeed()
    {
        return this.Orientation.worldToLocalMatrix * this.body.linearVelocity;
    }    

    public Vector2 GetDirection()
    {
        var fwd = this.transform.right;
        var dir = 
            new Vector2(){
                x = fwd.x,
                y = fwd.z
            }
            .normalized;

        return dir;
    }

    public Vector3 GetUpOrientation()
    {
        return this.Orientation.up;
    }
    public (Vector2 Direction, float Distance) GetDifferenceToTargetInLocalCoordinates()
    {
        var diff = this.Target.gameObject.transform.position - this.transform.position;
        var distance = diff.magnitude;

        var localY = Vector3.Dot(this.Orientation.forward, diff);
        var localX = Vector3.Dot(this.Orientation.right, diff);
        var localDiff = new Vector2(localX, localY);
        Debug.Log(localDiff);
        var localDiffDir = localDiff.normalized;

        return (localDiffDir, distance);
    }

    public float GetDistanceToFloor()
    {
        if (Physics.Raycast(this.Orientation.position, Vector3.down, out RaycastHit hit, float.PositiveInfinity, LayerMask.GetMask("Floor")))
            return hit.distance;
        else
            return float.PositiveInfinity;
    }


    Vector3 previousLinearVelocity;
    Vector3 previousAngularVelocity;

    Vector3 linearAcceleration;
    Vector3 angularAcceleration;

    public void FixedUpdate()
    {
        var linearVelocity = transform.InverseTransformVector(  this.body.linearVelocity);
        linearAcceleration = (linearVelocity - previousLinearVelocity) / Time.fixedDeltaTime;
        previousLinearVelocity = linearVelocity;

        var angularVelocity = transform.InverseTransformVector(this.body.angularVelocity);
        angularAcceleration = (angularVelocity - previousAngularVelocity) / Time.fixedDeltaTime;
        previousAngularVelocity = angularVelocity;
    }

    public HeadSensorsReading GetReading()
    {
        (var TargetDir, var TargetDistance) = this.GetDifferenceToTargetInLocalCoordinates();

        return new HeadSensorsReading {
            FloorDist = this.GetDistanceToFloor(),
            LocalTargetDir = TargetDir,
            TargetDist = TargetDistance,
            UpOrientation = this.GetUpOrientation(),
            LocalLinearSpeed = this.GetSpeed(),
            LocalAngularSpeed = this.previousLinearVelocity,
            LocalLinearAcceleration = this.linearAcceleration,
            LocalAngularAcceleration = this.angularAcceleration,
        };
    }
    // Update is called once per frame
    void Update()
    {
            
    }
}

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
public struct SerdeVector3
{
    public float x;
    public float y;
    public float z;


    public static implicit operator Vector3(SerdeVector3 a) => new Vector3(a.x, a.y, a.z);
    public static implicit operator SerdeVector3(Vector3 a) => new SerdeVector3{x = a.x,y =  a.y,z =  a.z};
    public override String ToString() => $"({x}, {y}, {z})";
}

[Serializable]
public struct HeadSensorsReading
{
    public SerdeVector2     LocalTargetDir;
    public float            TargetDist;
    public SerdeVector3     UpOrientation;
    public float            FloorDist;


    public SerdeVector3     LocalLinearSpeed;
    public SerdeVector3     LocalLinearAcceleration;

    public SerdeVector3     LocalAngularSpeed;
    public SerdeVector3     LocalAngularAcceleration;
    public override String ToString() => 
$@"HeadSensorsReading {{ 
    LocalTargetDir: {LocalTargetDir},
    TargetDist: {TargetDist},
    UpOrientation: {UpOrientation},
    FloorDist: {FloorDist},
    LocalSpeed: {LocalLinearSpeed},
    LocalLinearAcceleration:  {LocalLinearAcceleration},
    LocalAngularAcceleration: {LocalAngularAcceleration},
}}";
}
