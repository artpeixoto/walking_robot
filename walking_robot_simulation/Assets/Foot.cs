using UnityEngine;

public class Foot : MonoBehaviour
{
    Collider col;
    ArticulationBody body;
    void Start()
    {
        col = this.GetComponent<Collider>();
        body = this.GetComponent<ArticulationBody>();
    }
    public bool IsTouchingGround() => isTouchingGround;

    bool isTouchingGround;
    Vector3 floorForce;
    public Vector3 GetForceAppliedByFloor()
    {
        return this.floorForce;
    }
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.layer == LayerMask.NameToLayer("Floor"))
        {
            floorForce = collision.impulse;
            Debug.Log($"Floor force is {floorForce}");
            isTouchingGround = true;
        }
    }
    private void OnCollisionStay(Collision collision)
    {
        
    }
    private void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject.layer == LayerMask.NameToLayer("Floor"))
        {
            floorForce = Vector3.zero;
            isTouchingGround = false;
            Debug.Log($"Floor force is {floorForce}");
        }
    }
    // Update is called once per frame
    void FixedUpdate()
    {
    }
}
