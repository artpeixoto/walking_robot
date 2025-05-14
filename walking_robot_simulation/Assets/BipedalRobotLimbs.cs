using System;
using UnityEngine;

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

