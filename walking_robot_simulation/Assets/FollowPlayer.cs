using UnityEngine;

public class FollowPlayer : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    public Transform Player;
    public float Speed = 10.0f;
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        this.transform.position += Mathf.Lerp(0.0f, 1.0f, this.Speed * Time.deltaTime) * (this.Player.position - this.transform.position);
    }
}
