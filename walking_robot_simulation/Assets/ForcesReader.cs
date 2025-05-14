using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class ForcesReader : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    public Transform Orientation;
    public ForcesReadingsCollector Collector;
    
    void SendNoCollisionToCollector(){
        this.Collector.SetReadings(this, new List<ForceReading>());
    }

	void SendCollisionToCollector(Collision collision){
        var contacts = new List<ContactPoint>();
        collision.GetContacts(contacts);
        var forcesReadings = contacts.Select(c => {
            var position = this.Orientation.InverseTransformPoint(c.point);
            var force = this.Orientation.InverseTransformPoint(c.impulse);
            return new ForceReading{
                Force = force,
                Position = position
            };
        }).ToList();

        this.Collector.SetReadings(this,forcesReadings);
    }


    // Update is called once per frame
    void OnCollisionEnter(Collision collision)
    {
        this.SendCollisionToCollector(collision);             
    }
    void OnCollisionStay(Collision collision)
    {
        this.SendCollisionToCollector(collision); 
    }
    void OnCollisionExit(Collision collision)
    {
        this.SendCollisionToCollector(collision); 
    }
}

[Serializable]
public struct ForceReading{

    public SerdeVector3 Position;
    public SerdeVector3 Force;
}