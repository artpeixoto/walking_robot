using System;
using UnityEngine;

[Serializable]
public struct Limb
{
    public Foot  Foot;
    public Motor ShoulderMotor;
    public Motor ThighMotor;
    public Motor ShinMotor;

    public void ApplyActivation(LimbActivation input)
    {
        ShoulderMotor.SetInput(input.Shoulder);
        ThighMotor.SetInput(input.Thigh);
        ShinMotor.SetInput(input.Shin);
    }
    public LimbReading GetReading()
    {
        return new LimbReading {
            JointPositions = new LimbJointPositions {
                Shoulder = this.ShoulderMotor.GetMotorPosition(),
                Thigh = this.ThighMotor.GetMotorPosition(),
                Shin = this.ShinMotor.GetMotorPosition(),
            },
            JointSpeeds = new LimbJointSpeeds{
                Shoulder = this.ShoulderMotor.GetMotorSpeed(),
                Thigh = this.ThighMotor.GetMotorSpeed(),
                Shin = this.ShinMotor.GetMotorSpeed(),           
            },
            JointAccs = new LimbJointAccs{
                Shoulder = this.ShoulderMotor.GetMotorAcc(),
                Thigh = this.ThighMotor.GetMotorAcc(),
                Shin = this.ShinMotor.GetMotorAcc(),           
            },
            JointForces = new LimbJointForces {
                Shoulder = this.ShoulderMotor.GetJointForce(),
                Thigh = this.ThighMotor.GetJointForce(),
                Shin = this.ShinMotor.GetJointForce(),
            },
            IsFootTouchingFloor = this.Foot.IsTouchingGround(),
            ForceAppliedByFloor = this.Foot.GetForceAppliedByFloor(),
        };
    }
}

[Serializable]
public struct BipedalLimbsActivation
{
    public LimbActivation Left;
    public LimbActivation Right;

    public override string ToString() =>
$@"LimbsActivation{{
    Left: {Left.ToString().Replace("\n", "\n\t")}
    Right: {Right.ToString().Replace("\n", "\n\t")}
}}";
}

[Serializable]
public struct LimbActivation
{
    public float Shoulder;
    public float Thigh;
    public float Shin;
    public override string ToString() =>
$@"LimbActivation{{
    Shoulder: {Shoulder},
    Thigh: {Thigh},
    Shin: {Shin},
}}";


}
[Serializable]
public struct BipedalLimbsReading
{
    public LimbReading Left;
    public LimbReading Right;
    public override string ToString()
=> @$"LimbsReading{{ 
    Left: {Left.ToString().Replace("\n", "\n\t")},
    Right: {Right.ToString().Replace("\n", "\n\t")}
}}";
}

[Serializable]
public struct LimbJointPositions
{
    public float Shoulder;
    public float Thigh;
    public float Shin;

    public override string ToString()
=> @$"LimbJointPositions{{ 
    Shoulder: {Shoulder.ToString().Replace("\n", "\n\t")},
    Thigh: {Thigh.ToString().Replace("\n", "\n\t")},
    Shin: {Thigh.ToString().Replace("\n", "\n\t")}
}}";
}
[Serializable]
public struct LimbJointSpeeds
{
    public float Shoulder;
    public float Thigh;
    public float Shin;

    public override string ToString()
=> @$"LimbJointSpeeds{{ 
    Shoulder: {Shoulder.ToString().Replace("\n", "\n\t")},
    Thigh: {Thigh.ToString().Replace("\n", "\n\t")},
    Shin: {Shin.ToString().Replace("\n", "\n\t")}
}}";
}

[Serializable]
public struct LimbJointAccs{
    public float Shoulder;
    public float Thigh;
    public float Shin;

    public override string ToString()
=> @$"LimbJointAccs{{ 
    Shoulder: {Shoulder.ToString().Replace("\n", "\n\t")},
    Thigh: {Thigh.ToString().Replace("\n", "\n\t")},
    Shin: {Shin.ToString().Replace("\n", "\n\t")}
}}";
}

[Serializable]
public struct LimbJointForces
{
    public float Shoulder;
    public float Thigh;
    public float Shin;
    public override string ToString()
=> @$"LimbJointForces{{ 
    Shoulder: {Shoulder.ToString().Replace("\n", "\n\t")},
    Thigh: {Thigh.ToString().Replace("\n", "\n\t")},
    Shin: {Thigh.ToString().Replace("\n", "\n\t")}
}}";
}

[Serializable]
public struct LimbReading
{
    public LimbJointPositions   JointPositions;
    public LimbJointSpeeds      JointSpeeds;
    public LimbJointAccs        JointAccs;
    public LimbJointForces      JointForces;

    public bool  IsFootTouchingFloor;
    public SerdeVector3 ForceAppliedByFloor;

    public override String ToString() =>
$@"LimbReading{{
    JointPositions: {this.JointPositions.ToString().Replace("\n", "\n\t")},
    JointForces: {this.JointForces.ToString().Replace("\n", "\n\t")},
    IsFootTouchingFloor: {this.IsFootTouchingFloor},
    ForceAppliedByFloor: {this.ForceAppliedByFloor},
}}";
         
    //public Vector3 FootPosition;

}

public class BipedalRobotLimbs : MonoBehaviour
{
    public Limb LeftLimb;
    public Limb RightLimb;
        
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }
    public BipedalLimbsReading GetLimbReadings()
    {
        return new BipedalLimbsReading() {
            Left = this.LeftLimb.GetReading(),
            Right = this.RightLimb.GetReading()
        }; 
    }
    public void ApplyActivations(BipedalLimbsActivation acts)
    {
        this.LeftLimb.ApplyActivation(acts.Left);
        this.RightLimb.ApplyActivation(acts.Right);
    }

    // Update is called once per frame
    void Update()
    {
         
    }
}


public struct TransformData{

    public Vector3 LinearPos;
    public Vector3 LinearSpeed; 
    public Vector3 LinearAcc;

    public Vector3 AngularAcc;
    public Vector3 AngularSpeed;
    public Vector3 AngularPos;

}