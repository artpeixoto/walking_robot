using System;
using System.Collections.Generic;
using System.Security.Cryptography.X509Certificates;
using UnityEngine;

public class RobotHeadSensors : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    ArticulationBody body;
    ForcesReadingsCollector forcesCollector;

    public Target Target;
    public Transform Orientation;


    void Start()
    {
        this.body = this.GetComponent<ArticulationBody>();
        this.forcesCollector = this.GetComponent<ForcesReadingsCollector>();
    }

    public Vector3 GetSpeed()
    {
        return this.Orientation.InverseTransformVector(this.body.linearVelocity);
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

    public float GetDistanceToFloor()
    {
        if (Physics.Raycast(this.Orientation.position, Vector3.down, out RaycastHit hit, float.PositiveInfinity, LayerMask.GetMask("Floor")))
            return hit.distance;
        else
            return float.PositiveInfinity;
    }
    public Vector3 GetTargetPos(){
        return this.Orientation.InverseTransformPoint(this.Target.transform.position);
    }

    Vector3 linearVelocity;
    Vector3 angularVelocity;

    Vector3 linearAcceleration;
    Vector3 angularAcceleration;

    public void FixedUpdate()
    {
        var newLinearVelocity = transform.InverseTransformVector(  this.body.linearVelocity);
        linearAcceleration = (newLinearVelocity - this.linearVelocity) / Time.fixedDeltaTime;
        linearVelocity = newLinearVelocity;

        var newAngularVelocity = transform.InverseTransformVector(this.body.angularVelocity);
        angularAcceleration = (newAngularVelocity - angularVelocity) / Time.fixedDeltaTime;
        angularVelocity = newAngularVelocity;
    }

    public SensorsReading GetReading()
    {
        return new SensorsReading {
            FloorDist               = this.GetDistanceToFloor(),
            TargetPos               = this.GetTargetPos(),
            AccelerometerReading    = new AccelerometerReading{
                    UpOrientation       = this.GetUpOrientation(),
                    LinearSpeed         = this.GetSpeed(),
                    AngularSpeed        = this.linearVelocity,
                    LinearAcc  = this.linearAcceleration,
                    AngularAcc = this.angularAcceleration,
            },
            Forces   = this.forcesCollector.AllForcesReadings
        };
    }
}

[Serializable]
public struct SensorsReading
{
    public SerdeVector3         TargetPos;
    public float                FloorDist;
    public AccelerometerReading AccelerometerReading;

    public List<ForceReading> Forces;
}

[Serializable]
public struct AccelerometerReading{
    public SerdeVector3     LinearSpeed;
    public SerdeVector3     LinearAcc;

    public SerdeVector3     AngularSpeed;
    public SerdeVector3     AngularAcc;

    public SerdeVector3     UpOrientation;
}