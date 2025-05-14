using System;
using UnityEngine;

public class LimbLink : MonoBehaviour
{
    public Motor Motor;
    public TransformReader TransformReader;
    void Start()
    {
        this.Motor = this.GetComponent<Motor>();
        this.TransformReader = this.GetComponent<TransformReader>();
    }

    public LimbLinkReading GetReading(){
        return new LimbLinkReading{
            MotorReading = this.Motor.GetReading(),
            TransformReading = this.TransformReader.GetReading()
        };
    }
}

[Serializable]
public struct LimbReading
{
    public LimbLinkReading ShoulderReading;
    public LimbLinkReading ThighReading;
    public LimbLinkReading ShinReading;
    public TransformReading FootReading;
}

[Serializable]
public struct LimbLinkReading
{
    public MotorReading MotorReading;
    public TransformReading TransformReading;
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
public class Limb 
{
    public LimbLink         Shoulder;
    public LimbLink         Thigh;
    public LimbLink         Shin;
    public TransformReader  Foot;

    public void ApplyActivation(LimbActivation input)
    {
        this.Shoulder.Motor.SetInput(input.Shoulder);
        this.Thigh.Motor.SetInput(input.Thigh);
        this.Shin.Motor.SetInput(input.Shin);
    }

    public LimbReading GetReading()
    {
        return new LimbReading{
            ShoulderReading     = this.Shoulder.GetReading(),
            ThighReading        = this.Thigh.GetReading(),
            ShinReading         = this.Shin.GetReading(),
            FootReading         = this.Foot.GetReading(),
        };
    }
}
