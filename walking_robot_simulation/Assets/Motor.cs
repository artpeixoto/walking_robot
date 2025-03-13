using System;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;
public class Motor : MonoBehaviour
{
    public MotorAxis Axis;
    public float MaxSpeed = 15;
    public bool Invert;

    [SerializeField]
    float Input;
    
    public float GetInput()
    {
        return this.Input;
    }
    public void SetInput(float value)
    {
        this.Input = value;
    }

    ArticulationBody body;
    ArticulationBody parent;
    void Start()
    {
        body = GetComponent<ArticulationBody>();
        parent = this.gameObject.transform.parent.GetComponent<ArticulationBody>();
    }

    public float GetMotorAcc(){
        Debug.Log($"Joint acceleration is {jointAcceleration}");
        return jointAcceleration;
    }

    public float GetMotorSpeed()
    {
        Debug.Log($"Joint speed is {jointSpeed}");
        return jointSpeed;
    }
    public float GetMotorPosition()
    {
        var jointPositions = this.jointPosition;
        Debug.Log($"JointPosition is [{jointPositions}]");
        return jointPositions;
    }

    public float GetJointForce()
    {
        //var accumulatedTorque = new List<float>();
        //body.GetAccumulatedTorque();
        Debug.Log($"AccumulatedTorque is [{this.forces}]");
        return forces[0];
    }
    
    void ApplyForces(float deltaTime)
    {
        this.body.SetDriveTarget(ArticulationDriveAxis.X , this.Input * this.MaxSpeed );
    }

    float jointPosition;
    float jointSpeed;
    float jointAcceleration;
    void UpdateJointStuff(){
        var newJointPosition = this.body.jointPosition[0];
        var newJointSpeed    = (newJointPosition - jointPosition )/ Time.fixedDeltaTime;
        var newJointAcc      = (newJointSpeed - jointSpeed) / Time.fixedDeltaTime;
        jointPosition = newJointPosition;
        jointSpeed= newJointSpeed;
        jointAcceleration = newJointAcc;
    }
    Vector3 forces;

    void FixedUpdate()
    {
        ApplyForces(Time.fixedDeltaTime);
        UpdateJointStuff();
        forces = this.body.GetAccumulatedForce();
    }
}

[Serializable]
public enum MotorAxis
{
    X,Y,Z
}
public static class MotorAxisExts
{
    public static Vector3 GetVect(this MotorAxis _this)
    {
        switch (_this)
        {
            case MotorAxis.X: return Vector3.right;
            case MotorAxis.Y: return Vector3.up;
            case MotorAxis.Z: return Vector3.forward;
            default: throw new NotImplementedException();
        }
    }
}

