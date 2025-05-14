using System;
using UnityEngine;

public class TransformReader : MonoBehaviour
{
    ArticulationBody body;
    public Transform Orientation;
    void Start()
    {
        this.body = this.GetComponent<ArticulationBody>();
        this.body.automaticCenterOfMass = true;
        this.body.automaticInertiaTensor = true;
    }

    [SerializeField]
	public TransformReading reading = default;
    public void UpdateTransformReading(float fixedDeltaTime){
        var currentLinearPosition   = this.Orientation.InverseTransformPoint(this.body.worldCenterOfMass);
        var currentLinearSpeed      = this.Orientation.InverseTransformVector(this.body.linearVelocity) ;
        var currentLinearAcc        = (currentLinearSpeed        - (Vector3)this.reading.LinearSpeed) / fixedDeltaTime;

        var currentAngularPosition  = Quaternion.Inverse(this.Orientation.rotation) * this.transform.rotation;
        var currentAngularSpeed     = this.Orientation.InverseTransformVector(this.body.angularVelocity);
        var currentAngularAcc       = (currentAngularSpeed - (Vector3) this.reading.AngularSpeed) / fixedDeltaTime;

        this.reading = new TransformReading{
            LinearAcc = currentLinearAcc,
            LinearSpeed = currentLinearSpeed,
            LinearPos = currentLinearPosition,            

            AngularAcc = currentAngularAcc,
            AngularSpeed = currentAngularSpeed,
            AngularPos = currentAngularPosition,
        };
    }
    public TransformReading GetReading(){
        return this.reading;
    }

    void FixedUpdate(){
        this.UpdateTransformReading(Time.fixedDeltaTime);
    }
}

[Serializable]
public struct TransformReading{

    public SerdeVector3 LinearPos;
    public SerdeVector3 LinearSpeed; 
    public SerdeVector3 LinearAcc;

    public SerdeVector3     AngularAcc;
    public SerdeVector3     AngularSpeed;
    public SerdeQuaternion  AngularPos;

}

