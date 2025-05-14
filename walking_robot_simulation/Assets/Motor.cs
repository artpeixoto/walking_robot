using System;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;
public class Motor : MonoBehaviour
{
    public float MaxSpeed = 15;
    public float Jerk = 300;
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
    public MotorReading GetReading(){
        return this.MotorReading;
    }

    ArticulationBody body;
    ArticulationBody parent;
    void Start()
    {
        body = GetComponent<ArticulationBody>();
        parent = this.gameObject.transform.parent.GetComponent<ArticulationBody>();
    }
    
    [SerializeField]
    float target = 0;
    void applyInput(float deltaTime)
    {
        var targetTarget = this.Input * this.MaxSpeed;
        var distanceToTarget = targetTarget - target;
        var targetMaxDelta = deltaTime * this.Jerk;
        float targetDelta = 
            distanceToTarget >= 0 
            ? Mathf.Min(targetMaxDelta, distanceToTarget) 
            : Mathf.Max(-targetMaxDelta, distanceToTarget);
        
        target += targetDelta;

        this.body.SetDriveTarget(ArticulationDriveAxis.X , target );
    }

    public MotorReading MotorReading;
    void updateMotorReading(){
        var newJointPosition    = this.body.jointPosition[0];
        var newJointSpeed       = (newJointPosition - MotorReading.Pos)/ Time.fixedDeltaTime;
        var newJointAcc         = (newJointSpeed - MotorReading.Speed) / Time.fixedDeltaTime;

		var driveForces = new List<float>();
        this.body.GetDriveForces(driveForces);
        var driveForce = driveForces[0];

        MotorReading = new MotorReading{
            Pos     = newJointPosition,
            Speed   = newJointSpeed,
            Acc     = newJointAcc,
        };
    }

    Vector3 forces;

    void FixedUpdate()
    {
        applyInput(Time.fixedDeltaTime);
        updateMotorReading();
        forces = this.body.GetAccumulatedForce();
    }
}

[Serializable]
public struct MotorReading
{
    public float Pos;
    public float Speed;
    public float Acc;
    public float Torque;
}


