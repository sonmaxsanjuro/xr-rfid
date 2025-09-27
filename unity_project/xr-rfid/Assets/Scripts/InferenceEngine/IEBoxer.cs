using System.Collections.Generic;
using UnityEngine;
using Unity.InferenceEngine;
using UnityEngine.UI;
using System.IO;
using UnityEngine.Networking;

public struct BoundingBox
{
    public float CenterX;
    public float CenterY;
    public float Width;
    public float Height;
    public string Label;
    public Vector3? WorldPos;
    public string ClassName;
}

public class IEBoxer : MonoBehaviour
{
    [SerializeField] private Transform _displayLocation;
    [SerializeField] private TextAsset _labelsAsset;
    [SerializeField] private Color _boxColor;
    [SerializeField] private Sprite _boxTexture;
    [SerializeField] private Font _font;
    [SerializeField] private Color _fontColor;
    [SerializeField] private int _fontSize = 80;


    // Added a new public field to specify the label to filter by
    [SerializeField] private string _targetLabel;


    private string[] _labels;
    private List<GameObject> _boxPool = new();


    private void Start()
    {
        _labels = _labelsAsset.text.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
        Debug.Log($"Loaded {_labels.Length} labels from {_labelsAsset.name}");
    }

    public void MakeSynchronousRequest(string url)
    {
        Debug.Log("Starting synchronous web request...");

        //const float timeoutSeconds = 0.1; // Set your desired timeout in seconds
        float startTime = Time.time;
        UnityWebRequest request = UnityWebRequest.Get(url);
        
        // Send the request and immediately wait for it to complete
        request.SendWebRequest();

        // Loop until the request is done. This will block the main thread.
        while (!request.isDone)
        {
            // You can add a timeout here to prevent infinite blocking
            // if the request never completes.
            if (Time.time > startTime + 0.1) { break; }
        }

        if (request.isDone)
        {
            if (request.result == UnityWebRequest.Result.ConnectionError ||
                    request.result == UnityWebRequest.Result.ProtocolError)
            {
                _targetLabel = "none";
                //Debug.LogError("Error: " + request.error);
            }
            else
            {
                Debug.Log("Received response: " + request.downloadHandler.text);
                _targetLabel = request.downloadHandler.text;
            }
        }
        else
        {
            request.Abort();
        }

        // Dispose of the request to free up resources
        request.Dispose();

        Debug.Log("Synchronous web request finished.");
    }
    public List<BoundingBox> DrawBoxes(Tensor<float> output, Tensor<int> labelIds, float imageWidth, float imageHeight)
    {
        List<BoundingBox> boundingBoxes = new();

        var scaleX = imageWidth / 640;
        var scaleY = imageHeight / 640;

        var halfWidth = imageWidth / 2;
        var halfHeight = imageHeight / 2;

        int boxesFound = output.shape[0];
        if (boxesFound <= 0) 
            return boundingBoxes;

        var maxBoxes = Mathf.Min(boxesFound, 200);
        int boxesDrawnCount = 0;

        //get rfid from pico
        string targetUrl = "raspberrypi.local";
        int targetPort = 5000;
        string url = $"http://{targetUrl}:{targetPort}/";

        MakeSynchronousRequest(url);


        //_targetLabel = "230";
        string _correctLabel = _targetLabel;
        string _compareLabel = _targetLabel;
        

        switch (_targetLabel)
        {
            //for longer harness and Sparkfun MiFare tags
            case "40865ef1":
                _correctLabel = "yellow";
                _targetLabel = "row 1, column 1";
                break;
            case "70855ef1":
                _correctLabel = "yellow";
                _targetLabel = "row 3, column 2";
                break;
            case "e0b95df1":
                _correctLabel = "blue";
                _targetLabel = "row 1, column 2";
                break;
            case "a0c760f1":
                _correctLabel = "blue";
                _targetLabel = "row 3, column 1";
                break;
            case "70805ef1":
                _correctLabel = "blue";
                _targetLabel = "row 5, column 2";
                break;
            case "407a5df1":
                _correctLabel = "red";
                _targetLabel = "row 1, column 2";
                break;
            case "40845ef1":
                _correctLabel = "red";
                _targetLabel = "row 3, column 1";
                break;
            case "907d5df1":
                _correctLabel = "black";
                _targetLabel = "row 1, column 1";
                break;
            case "c0bb5df1":
                _correctLabel = "black";
                _targetLabel = "row 2, column 2";
                break;
            case "90a35df1":
                _correctLabel = "black";
                _targetLabel = "row 3, column 1";
                break;
            //for shorter harness and nfc213 tags
            case "249":
                _correctLabel = "yellow";
                _targetLabel = "row 1, column 1";
                break;
            case "235":
                _correctLabel = "yellow";
                _targetLabel = "row 3, column 2";
                break;
            case "239":
                _correctLabel = "blue";
                _targetLabel = "row 1, column 2";
                break;
            case "243":
                _correctLabel = "blue";
                _targetLabel = "row 3, column 1";
                break;
            case "252":
                _correctLabel = "blue";
                _targetLabel = "row 5, column 2";
                break;
            case "230":
                _correctLabel = "red";
                _targetLabel = "row 1, column 2";
                break;
            case "240":
                _correctLabel = "red";
                _targetLabel = "row 3, column 1";
                break;
            case "251":
                _correctLabel = "black";
                _targetLabel = "row 1, column 1";
                break;
            case "224":
                _correctLabel = "black";
                _targetLabel = "row 2, column 2";
                break;
            case "247":
                _correctLabel = "black";
                _targetLabel = "row 3, column 1";
                break;
            default:
                _correctLabel = null;
                break;

        }

        Debug.Log("received label: " + _correctLabel + _targetLabel);        
        switch (_correctLabel)
        {
            case "black":
                _compareLabel = "yellow";
                break;
            case "red":
                _compareLabel = "black";
                break;
            case "yellow":
                _compareLabel = "red";
                break;
            case "blue":
                _compareLabel = "blue";
                break;
            default:
                _compareLabel = null;
                break;

        }

        Debug.Log("compare label: " + _compareLabel);        
        

        for (var n = 0; n < maxBoxes; n++)
        {
            // Get object class name
            var classname = _labels[labelIds[n]].Replace(" ", "_");

            // Check if the current box's label matches the target label
            if (classname.Equals(_compareLabel, System.StringComparison.OrdinalIgnoreCase))
            {
                Debug.Log("entering drawing boxes.");
                // Get bounding box center coordinates
                var centerX = output[n, 0] * scaleX - halfWidth;
                var centerY = output[n, 1] * scaleY - halfHeight;

                // Create a new bounding box
                var box = new BoundingBox
                {
                    CenterX = centerX,
                    CenterY = centerY,
                    ClassName = _correctLabel,
                    Width = output[n, 2] * scaleX,
                    Height = output[n, 3] * scaleY,
                    Label = $"{_correctLabel + " " + _targetLabel}",
                };

                Debug.Log($"Box {n}: {box.Label} - Center: ({box.CenterX}, {box.CenterY}), Size: ({box.Width}, {box.Height})");

                boundingBoxes.Add(box);

                DrawBox(box, boxesDrawnCount);
                boxesDrawnCount++;
                _targetLabel = null;
            }
        }

        ClearBoxes(boxesDrawnCount); 

        return boundingBoxes;
    }

    public void ClearBoxes(int lastBoxCount)
    {
        if (lastBoxCount < _boxPool.Count)
        {
            for (int i = lastBoxCount; i < _boxPool.Count; i++)
            {
                if (_boxPool[i] != null)
                {
                    _boxPool[i].SetActive(false);
                }
            }
        }
        //HideImage();
    }

    private void DrawBox(BoundingBox box, int id)
    {
        GameObject panel;
        if (id < _boxPool.Count)
        {
            panel = _boxPool[id];
            if (panel == null)
            {
                panel = CreateNewBox(_boxColor);
            }
            else
            {
                panel.SetActive(true);
            }
        }
        else
        {
            panel = CreateNewBox(_boxColor);
        }

        // Set box position
        panel.transform.localPosition = new Vector3(box.CenterX, -box.CenterY, box.WorldPos.HasValue ? box.WorldPos.Value.z : 0.0f);

        // Set box size
        RectTransform rectTransform = panel.GetComponent<RectTransform>();
        rectTransform.sizeDelta = new Vector2(box.Width, box.Height);

        // Set label text
        Text label = panel.GetComponentInChildren<Text>();
        label.text = box.Label;
    }

    private GameObject CreateNewBox(Color color)
    {
        // Create the box and set image
        GameObject panel = new("ObjectBox");
        panel.AddComponent<CanvasRenderer>();

        Image image = panel.AddComponent<Image>();
        image.color = color;
        image.sprite = _boxTexture;
        image.type = Image.Type.Sliced;
        image.fillCenter = false;
        panel.transform.SetParent(_displayLocation, false);

        // Create the label
        GameObject textGameObject = new("ObjectLabel");
        textGameObject.AddComponent<CanvasRenderer>();
        textGameObject.transform.SetParent(panel.transform, false);

        Text text = textGameObject.AddComponent<Text>();
        text.font = _font;
        text.color = _fontColor;
        text.fontSize = _fontSize;
        text.horizontalOverflow = HorizontalWrapMode.Overflow;

        RectTransform rectTransform = textGameObject.GetComponent<RectTransform>();
        rectTransform.offsetMin = new Vector2(20, rectTransform.offsetMin.y);
        rectTransform.offsetMax = new Vector2(0, rectTransform.offsetMax.y);
        rectTransform.offsetMin = new Vector2(rectTransform.offsetMin.x, 0);
        rectTransform.offsetMax = new Vector2(rectTransform.offsetMax.x, 30);
        rectTransform.anchorMin = new Vector2(0, 0);
        rectTransform.anchorMax = new Vector2(1, 1);

        _boxPool.Add(panel);

        return panel;
    }

}
