using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class ForcesReadingsCollector : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        this.ForceReadings = new Dictionary<ForcesReader, List<ForceReading>>();    
    }
    public List<ForceReading> GetReadings(){
        var readings = this.ForceReadings.SelectMany(kvp => kvp.Value).ToList();
        return readings;
    }


	public void SetReadings(ForcesReader reader, List<ForceReading> readings){
        this.ForceReadings[reader] = readings;
    }
    public void LateUpdate()
    {
        this.AllForcesReadings = this.GetReadings();
    }
    public List<ForceReading> AllForcesReadings;

    [SerializeField]
	public Dictionary<ForcesReader, List<ForceReading>> ForceReadings;
}
