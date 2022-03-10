using System.Collections;
using System.Text;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;

public class Dgame : MonoBehaviour
{
    bool playerTurn = true;
    bool AiThinking = false;
    int redPieces = 12;
    int whitePieces = 12;
    private Vector3 scaleChange;

    public bool mandatory_jumping = false;
    private float offsetx = -4;
    private float offsetz = -4;
    List<GameObject> moveSpots = new List<GameObject>();
    List<GameObject> playerPieces = new List<GameObject>();
    List<GameObject> aiPieces = new List<GameObject>();
    GameObject currentPiece;


    [SerializeField] public int[,] board = new int[8, 8];
    void createPiece(int n, int m, int oddRow = 0, bool isBrown = true)
    {
        GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        cylinder.transform.position = new Vector3(0 + 2 * n + offsetx + oddRow, 0.5f, m + offsetz);
        board[(int)(m + offsetz + 4f), (int)(0 + 2 * n + offsetx + oddRow + 4f)] = 1;
        scaleChange = new Vector3(0.8f, 0.2f, 0.8f);
        cylinder.transform.localScale = scaleChange;

        if (isBrown)
        {
            cylinder.GetComponent<Renderer>().material.color = new Color32(147, 48, 48, 255);
            cylinder.name = "1";
            playerPieces.Add(cylinder);
            board[(int)(m + offsetz + 4f), (int)(0 + 2 * n + offsetx + oddRow + 4f)] = 1;        // player
        }

        else
        {
            // aIPieces.Add(((int)(m + offsetz + 4f)).ToString() + ((int)(0 + 2 * n + offsetx + oddRow + 4f)).ToString(), cylinder);
            /*Debug.Log(((int)(m)).ToString() + ((int)(0 + 2 * n  + oddRow)).ToString());*/
            cylinder.name = "2";
            aiPieces.Add(cylinder);
            board[(int)(m + offsetz + 4f), (int)(0 + 2 * n + offsetx + oddRow + 4f)] = 2;
            cylinder.GetComponent<Renderer>().material.color = Color.white;
        }



    }
    void createBoard(int row, int col, bool isRed = false)
    {
        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.transform.position = new Vector3(row + offsetx, 0, col + offsetz);

        if (isRed)
            cube.GetComponent<Renderer>().material.color = Color.red;
        else
            cube.GetComponent<Renderer>().material.color = Color.black;
    }
    void setGame()
    {
        for (int col = 0; col < 8; col++)
        {
            for (int row = 0; row < 8; row++)
            {
                if ((col + row) % 2 == 1)
                    createBoard(row, col, true);
                else
                    createBoard(row, col, false);
                board[col, row] = 0;
            }

        }

        for (int col = 0; col < 4; col++)
        {
            for (int row = 0; row < 3; row++)
            {
                if (row % 2 == 1) createPiece(col, row, 1);
                else createPiece(col, row);
            }

        }

        for (int col = 0; col < 4; col++)
        {
            for (int row = 5; row < 8; row++)
            {
                if (row % 2 == 1)
                    createPiece(col, row, 1, false);
                else
                    createPiece(col, row, 0, false);

            }

        }
    }

    void printBoard()
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < board.GetLength(1); i++)
        {
            for (int j = 0; j < board.GetLength(0); j++)
            {
                sb.Append(board[7 - i, j]);
                sb.Append(' ');
            }
            sb.AppendLine();
        }
        Debug.Log(sb.ToString());
    }

 public GameObject GetItemAtCoordinate(int x, int y) 
 {
     foreach (GameObject g in playerPieces)
     {
         if (g && Mathf.RoundToInt(g.transform.position.x) == y +offsetx && Mathf.RoundToInt(g.transform.position.z) == x +offsetz)
         {
             return g;
         }
     }

     foreach (GameObject g in aiPieces)
     {
         if (g && Mathf.RoundToInt(g.transform.position.x) == y +offsetx && Mathf.RoundToInt(g.transform.position.z) == x +offsetz)
         {
             return g;
         }
     }
     return null;
 }

    public static (int i_cord, int j_cord) getBoardCoordinates(GameObject piece)
    {
        Vector3 current_piece = piece.transform.position;
        int x = (int)current_piece.x;
        int y = (int)current_piece.z;

        return ((y + 4), x + 4);

    }

    GameObject debugpiece()
    {
        if (Input.GetMouseButtonDown(0))
        {
            RaycastHit hit;
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);

            if (Physics.Raycast(ray,out hit))
            {
                return hit.transform.gameObject;
            }
        }
        return null;
    }
    
public List<List<int>> findPlayerAvailableMoves(int[,] board, bool mandatory_jumping)
    {
        List<List<int>> available_moves = new List<List<int>>();
        List<List<int>> available_jumps = new List<List<int>>();

        for (int m = 0; m < 8; m++)
        {
            for (int n = 0; n < 8; n++)
            {
                if (board[m, n] == 1)
                {
                    /*Debug.Log(m.ToString() + " : " + n.ToString());*/
                    if (checkPlayerMoves(board, m, n, m + 1, n + 1))
                        available_moves.Add(new List<int> { m, n, m + 1, n + 1 });
                    if (checkPlayerMoves(board, m, n, m + 1, n - 1))
                        available_moves.Add(new List<int> { m, n, m + 1, n - 1 });
                    if (checkPlayerJumps(board, m, n, m + 1, n - 1, m + 2, n - 2))
                        available_jumps.Add(new List<int> { m, n, m + 2, n - 2 });
                    if (checkPlayerJumps(board, m, n, m + 1, n + 1, m + 2, n + 2))
                        available_jumps.Add(new List<int> { m, n, m + 2, n + 2 });
                }
                else if (board[m, n] == 3)
                {
                    if (checkPlayerMoves(board, m, n, m + 1, n + 1))
                        available_moves.Add(new List<int> { m, n, m + 1, n + 1 });
                    if (checkPlayerMoves(board, m, n, m + 1, n - 1))
                        available_moves.Add(new List<int> { m, n, m + 1, n - 1 });
                    if (checkPlayerMoves(board, m, n, m - 1, n - 1))
                        available_moves.Add(new List<int> { m, n, m - 1, n - 1 });
                    if (checkPlayerMoves(board, m, n, m - 1, n + 1))
                        available_moves.Add(new List<int> { m, n, m - 1, n + 1 });
                    if (checkPlayerJumps(board, m, n, m + 1, n - 1, m + 2, n - 2))
                        available_jumps.Add(new List<int> { m, n, m + 2, n - 2 });
                    if (checkPlayerJumps(board, m, n, m - 1, n - 1, m - 2, n - 2))
                        available_jumps.Add(new List<int> { m, n, m - 2, n - 2 });
                    if (checkPlayerJumps(board, m, n, m - 1, n + 1, m - 2, n + 2))
                        available_jumps.Add(new List<int> { m, n, m - 2, n + 2 });
                    if (checkPlayerJumps(board, m, n, m + 1, n + 1, m + 2, n + 2))
                        available_jumps.Add(new List<int> { m, n, m + 2, n + 2 });
                }
            }
        }

        if (!mandatory_jumping)
        {
            available_jumps.AddRange(available_moves);
            return available_jumps;
        }
        else
        {
            if (available_jumps.Count == 0)
                return available_moves;
            else
                return available_jumps;
        }

    }

    bool checkPlayerMoves(int[,] board, int old_i, int old_j, int new_i, int new_j)
    {
        /*Debug.Log(old_i.ToString() + old_j.ToString() + new_i.ToString() + new_j.ToString());*/
        if (new_i > 7 || new_i < 0)
            return false;
        if (new_j > 7 || new_j < 0)
            return false;
        if (board[old_i, old_j] == 0)
            return false;
        if (board[new_i, new_j] != 0)
            return false;
        if (board[old_i, old_j] == 2 || board[old_i, old_j] == 4)
            return false;
        if (board[new_i, new_j] == 0)
            return true;
        return true;
    }

    bool checkPlayerJumps(int[,] board, int old_i, int old_j, int via_i, int via_j, int new_i, int new_j)
    {
        if (new_i > 7 || new_i < 0)
            return false;
        if (new_j > 7 || new_j < 0)
            return false;
        if (board[via_i, via_j] == 0)
            return false;
        if (board[via_i, via_j] == 1 || board[via_i, via_j] == 3)
            return false;
        if (board[new_i, new_j] != 0)
            return false;
        if (board[old_i, old_j] == 0)
            return false;
        if (board[old_i, old_j] == 2 || board[old_i, old_j] == 4)
            return false;
        return true;
    }

    void getCurrentAvailableMoves(int currentpiece_x, int currentpiece_y)
    {
        // List<int> currentPieceMovies = new List<int>();
        List<List<int>> moves = findPlayerAvailableMoves(board, mandatory_jumping);
        foreach (var move in moves)
        {
            if (move[0] == currentpiece_x && move[1] == currentpiece_y)
            {
                /*currentPieceMovies.Add(move);*/
                GameObject movespot = GameObject.CreatePrimitive(PrimitiveType.Plane);
                movespot.transform.position = new Vector3(move[3] + offsetx, 0.55f, move[2] + offsetz);
                movespot.name = "movespot";
                movespot.transform.localScale = new Vector3(0.1f, 0.05f, 0.1f);
                moveSpots.Add(movespot);
            }
        }
    }

    void makeaMove(int old_i, int old_j, int new_i, int new_j, int queen_row)
    {
        

        int player;
        if (board[old_i, old_j] == 3)
        {
            player = 3;
        }
        else if(board[old_i, old_j] == 4)
        {
            player = 4;
        }
        else
        {
            player = (board[old_i, old_j] % 2);
        }
        
        
        if (player == 0)
            player = 2;
        int i_difference = old_i - new_i;
        int j_difference = old_j - new_j;



        if (i_difference == -2 && j_difference == 2)
        {
            board[old_i + 1, old_j - 1] = 0;
            Destroy(GetItemAtCoordinate(old_i + 1, old_j - 1));
            updatePiecesCount(old_i + 1, old_j - 1);
        }

        else if (i_difference == 2 && j_difference == 2)
        {
            board[old_i - 1, old_j - 1] = 0;
            Destroy(GetItemAtCoordinate(old_i - 1, old_j - 1));
            updatePiecesCount(old_i - 1, old_j - 1);
        }


        else if (i_difference == 2 && j_difference == -2)
        {
            board[old_i - 1, old_j + 1] = 0;
            Destroy(GetItemAtCoordinate(old_i - 1, old_j + 1));
            updatePiecesCount(old_i - 1, old_j + 1);
        }

        else if (i_difference == -2 && j_difference == -2)
        {
            board[old_i + 1, old_j + 1] = 0;
            Destroy(GetItemAtCoordinate(old_i + 1, old_j + 1));
            updatePiecesCount(old_i + 1, old_j + 1);
        }


        if (new_i == queen_row)
        {
            if(player < 3)
            {
                player += 2;              // Promotion to King
                GameObject piece = GetItemAtCoordinate(old_i, old_j);
                piece.transform.localScale = new Vector3(0.8f, 1.0f, 0.8f);
            }
            
        }

        GameObject movea = GetItemAtCoordinate(old_i, old_j);
        Debug.Log(movea);
        if(movea)
            movea.transform.position = new Vector3(new_j + offsetx, 0.5f, new_i + offsetz);
        Debug.Log(old_i + " " + old_j + " " + new_i + " " + new_j);
        board[old_i, old_j] = 0;
        board[new_i, new_j] = player;
        // foreach (GameObject x in moveSpots)
        // {
        //     Destroy(x);
        // }
        Debug.Log("Move has been made");
        printBoard();
    }

    void updatePiecesCount(int x, int y)
    {
        if (GetItemAtCoordinate(x, y).GetComponent<Renderer>().material.color == Color.white)
            whitePieces--;
        else
            redPieces--;

    }

    bool checkWin()
    {
        if (whitePieces * redPieces == 0)
            return true;
        return false;
    }

    void playerMove(){
            GameObject selectedPiece = debugpiece();

            if(selectedPiece){
                if( selectedPiece.name=="1"){
                    // foreach(GameObject movespot in moveSpots){
                    //     Destroy(movespot);
                    // }
                    currentPiece = selectedPiece;
                    int i,j;
                    (i,j) = getBoardCoordinates(selectedPiece);
                    getCurrentAvailableMoves( i, j);
                }else if(selectedPiece.name=="movespot"){
                    int nx, ny, px, py;
                    (px,py) = getBoardCoordinates(currentPiece);
                    (nx,ny) = getBoardCoordinates(selectedPiece);
                    makeaMove( px, py, nx, ny, 7);
                    Debug.Log("Move has been made");
                    playerTurn = false;
                    foreach(GameObject movespot in moveSpots){
                        Destroy(movespot);
                    }
                }
            
            }
        
    }

    void aIMove(){
        
        // StartCoroutine(PostRequest("http://127.0.0.1:5000/", fin));

        if(gameObject.GetComponent<Server>().move.Length>0){
            string[] cmove = gameObject.GetComponent<Server>().move;
            int px, py, nx, ny;
            px = int.Parse(cmove[0]);
            py = int.Parse(cmove[1]);
            nx = int.Parse(cmove[2]);
            ny = int.Parse(cmove[3]);
            Debug.Log(px + " " + py + " " + nx + " " + ny);
            makeaMove( px, py, nx, ny, 0);
            gameObject.GetComponent<Server>().move = new string[0];
            AiThinking = false;
            playerTurn = true;
        }else if(!AiThinking){
            gameObject.GetComponent<Server>().PostData();
            AiThinking = true;
        }

        
    }

    void Start()
    {
        Debug.Log("Game Started");
        setGame();
        printBoard();
    }

    void Update()
    {
        bool temp = checkWin();
        if(temp){
            Debug.Log("Game Over!");
            Application.Quit();
        }
        else{
            if(playerTurn){
            playerMove();
            }else{
                aIMove();
            }
        }
        
    }
}
