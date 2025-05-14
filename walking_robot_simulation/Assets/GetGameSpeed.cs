using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class GetGameSpeed : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    public TextMeshProUGUI text;
    void Start()
    {
        text = GetComponent<TextMeshProUGUI>(); 
    }

    // Update is called once per frame
    void Update()
    {
        text.text = $"{Time.timeScale:0.00}"; 
    }
}
