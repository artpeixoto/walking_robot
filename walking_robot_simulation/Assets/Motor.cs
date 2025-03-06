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


    public float GetMotorPosition()
    {
        var jointPositions = this.body.jointPosition;
        Debug.Log($"JointPosition is [{jointPositions[0]}]");
        return this.body.jointPosition[ 0 ];
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
    Vector3 forces;
    void FixedUpdate()
    {
        ApplyForces(Time.fixedDeltaTime);
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

