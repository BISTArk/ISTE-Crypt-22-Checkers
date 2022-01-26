using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
// using UnityEngine.UI;

public class Server : MonoBehaviour {

    // public InputField userID;
    // public InputField userName;
    // public InputField score;

    // [Serializable]
    // public class Gamer
    // {
    //     public int userId;
    //     public string userName;
    //     public int score;
    // }
    public String[] move;

    public void PostData()
    {
        int[,] Grid = gameObject.GetComponent<Dgame>().board;
        Debug.Log("PostData");
        string fin = "";
        for(int i = 0; i < 8; i++)
            {
                string data = "";
                for(int j = 0; j < 8; j++){
                    data +=Grid[i,j].ToString() + " ";
                }
                fin=fin+data+"\n";
                //or
            }
        Debug.Log(fin);

        //convert 2d matrix to string
        StartCoroutine(PostRequest("http://127.0.0.1:5000/", fin));
    }

    IEnumerator PostRequest(string url, string json)
    {
        var uwr = new UnityWebRequest(url, "POST");
        byte[] jsonToSend = new System.Text.UTF8Encoding().GetBytes(json);
        uwr.uploadHandler = (UploadHandler)new UploadHandlerRaw(jsonToSend);
        uwr.downloadHandler = (DownloadHandler)new DownloadHandlerBuffer();
        uwr.SetRequestHeader("Content-Type", "application/json");

        //Send the request then wait here until it returns
        yield return uwr.SendWebRequest();
        if (uwr.result == UnityWebRequest.Result.ConnectionError)
        {
            Debug.Log("Error While Sending: " + uwr.error);
        }
        else
        {
            String[] spearator = { " " };
            String stri = uwr.downloadHandler.text;
            move = stri.Split(spearator, 4, StringSplitOptions.RemoveEmptyEntries);
            foreach (String x in move)
            {
                Debug.Log(x);
            }
            Debug.Log("Received: " + uwr.downloadHandler.text);
        }
    }
}