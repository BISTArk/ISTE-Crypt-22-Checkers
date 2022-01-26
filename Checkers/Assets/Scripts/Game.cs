using System.Collections;
using System.Text;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;

public class Game : MonoBehaviour
{
    // Start is called before the first frame update
    private Vector3 scaleChange;
    int redPieces = 12;
    int whitePieces = 12;
    private float offsetx = -4;
    private float offsetz = -4;
    int currentpiece_x;
    int currentpiece_y;
    string currentState = "PlayerToSelect";
    GameObject currentPiece;
    List<GameObject> moveSpots = new List<GameObject>();
    Hashtable aIPieces = new Hashtable();

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
            board[(int)(m + offsetz + 4f), (int)(0 + 2 * n + offsetx + oddRow + 4f)] = 1;        // player
        }
            
        else
        {
            aIPieces.Add( ((int)(m + offsetz + 4f)).ToString() + ((int)(0 + 2 * n + offsetx + oddRow + 4f)).ToString(), cylinder);
            /*Debug.Log(((int)(m)).ToString() + ((int)(0 + 2 * n  + oddRow)).ToString());*/
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
                if (row % 2 == 1)
                    createPiece(col, row, 1);
                else
                    createPiece(col, row);


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

    public List<List<int>> findAvailableMoves(int[,] board, bool mandatory_jumping)
    {
        List<List<int>> available_moves = new List<List<int>>();
        List<List<int>> available_jumps = new List<List<int>>();

        for (int m = 0; m < 8; m++)
        {
            for (int n = 0; n < 8; n++)
            {
                if (board[m, n] == 2)
                {
                    if (checkMoves(board, m, n, m - 1, n - 1))
                        available_moves.Add(new List<int> { m, n, m - 1, n - 1 });
                    if (checkMoves(board, m, n, m - 1, n + 1))
                        available_moves.Add(new List<int> { m, n, m - 1, n + 1 });
                    if (checkJumps(board, m, n, m - 1, n - 1, m - 2, n - 2))
                        available_jumps.Add(new List<int> { m, n, m - 2, n - 2 });
                    if (checkJumps(board, m, n, m - 1, n + 1, m - 2, n + 2))
                        available_jumps.Add(new List<int> { m, n, m - 2, n + 2 });
                }
                else if (board[m, n] == 4)
                {
                    if (checkMoves(board, m, n, m - 1, n - 1))
                        available_moves.Add(new List<int> { m, n, m - 1, n - 1 });
                    if (checkMoves(board, m, n, m - 1, n + 1))
                        available_moves.Add(new List<int> { m, n, m - 1, n + 1 });
                    if (checkJumps(board, m, n, m - 1, n - 1, m - 2, n - 2))
                        available_jumps.Add(new List<int> { m, n, m - 2, n - 2 });
                    if (checkJumps(board, m, n, m - 1, n + 1, m - 2, n + 2))
                        available_jumps.Add(new List<int> { m, n, m - 2, n + 2 });
                    if (checkMoves(board, m, n, m + 1, n - 1))
                        available_moves.Add(new List<int> { m, n, m + 1, n - 1 });
                    if (checkJumps(board, m, n, m + 1, n - 1, m + 2, n - 2))
                        available_jumps.Add(new List<int> { m, n, m + 2, n - 2 });
                    if (checkMoves(board, m, n, m + 1, n + 1))
                        available_moves.Add(new List<int> { m, n, m + 1, n + 1 });
                    if (checkJumps(board, m, n, m + 1, n + 1, m + 2, n + 2))
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

    bool checkMoves(int[,] board, int old_i, int old_j, int new_i, int new_j)
    {
        if (new_i > 7 || new_i < 0)
            return false;
        if (new_j > 7 || new_j < 0)
            return false;
        if (board[old_i, old_j] == 0)
            return false;
        if (board[new_i, new_j] != 0)
            return false;
        if (board[old_i, old_j] == 1 || board[old_i, old_j] == 3)
            return false;
        if (board[new_i, new_j] == 0)
            return true;
        return true;
    }

    bool checkJumps(int[,] board, int old_i, int old_j, int via_i, int via_j, int new_i, int new_j)  // check the function
    {
        if (new_i > 7 || new_i < 0)
            return false;
        if (new_j > 7 || new_j < 0)
            return false;
        if (board[via_i, via_j] == 0)
            return false;
        if (board[via_i, via_j] == 2 || board[via_i, via_j] == 4)
            return false;
        if (board[new_i, new_j] != 0)
            return false;
        if (board[old_i, old_j] == 0)
            return false;
        if (board[old_i, old_j] == 1 || board[old_i, old_j] == 3)
            return false;
        return true;
    }

    GameObject getGameObject(int x, int y)
    {
        RaycastHit hit;
        Vector3 camPos = Camera.main.transform.position;
        Ray ray = new Ray(camPos, new Vector3(y + offsetx, 0.5f, x + offsetz) - camPos);
        /*Debug.Log("Ray: " + (y).ToString() + " " + (x).ToString());*/
        GameObject piece = new GameObject();
        if (Physics.Raycast(ray, out hit, 100.0f))
        {
            if (hit.transform != null)
            {

                
                if (piece = hit.collider.gameObject)
                {
                    if (piece.name == "Cylinder")
                    {
                        Debug.Log("Piece");
                        int i, j;
                        (i, j) = getBoardCoordinates(piece);
                        Debug.Log(i.ToString() + " " + j.ToString());

                        /*piece.transform.position = new Vector3(move[3] + offsetx, 0.5f, move[2] + offsetz);*/

                    }
                    if (piece.name == "Plane")
                    {

                    }

                }

            }
            else
            {
                print("Ray is null");
            }


        }
        return piece;
    }

    public void makeAMove(int[,] board, int old_i, int old_j, int new_i, int new_j, string big_letter, int queen_row)
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
            Destroy(getGameObject(old_i + 1, old_j - 1));
            updatePiecesCount(old_i + 1, old_j - 1);
        }

        else if (i_difference == 2 && j_difference == 2)
        {
            board[old_i - 1, old_j - 1] = 0;
            Destroy(getGameObject(old_i - 1, old_j - 1));
            updatePiecesCount(old_i - 1, old_j - 1);
        }


        else if (i_difference == 2 && j_difference == -2)
        {
            board[old_i - 1, old_j + 1] = 0;
            Destroy(getGameObject(old_i - 1, old_j + 1));
            updatePiecesCount(old_i - 1, old_j + 1);
        }

        else if (i_difference == -2 && j_difference == -2)
        {
            board[old_i + 1, old_j + 1] = 0;
            Destroy(getGameObject(old_i + 1, old_j + 1));
            updatePiecesCount(old_i + 1, old_j + 1);
        }


        if (new_i == queen_row)
        {
            if(player < 3)
            {
                player += 2;              // Promotion to King
                GameObject piece = getGameObject(old_i, old_j);
                piece.transform.localScale = new Vector3(0.8f, 0.4f, 0.8f);
            }
            
        }
        
            //letter = big_letter[0];

        board[old_i, old_j] = 0;

        //board[new_i][new_j] = letter + str(new_i) + str(new_j);
        board[new_i, new_j] = player;
        /*board[new_i, new_j] = "";
        board[new_i, new_j] += letter;
        board[new_i, new_j] += (new_i + '0');
        board[new_i, new_j] += (new_j + '0');*/
        Debug.Log("Move has been made");
        Debug.Log(old_i.ToString() + old_j.ToString() + new_i.ToString() + new_j.ToString());
        boardPositionDebug();
    }

    public static (int i_cord, int j_cord) getCoordinates(GameObject piece)
    {
        Vector3 current_piece = piece.transform.position;
        int x = (int)current_piece.x;
        int y = (int)current_piece.z;

        return (7 - (y + 4), x + 4);

    }
    public static (int i_cord, int j_cord) getBoardCoordinates(GameObject piece)
    {
        Vector3 current_piece = piece.transform.position;
        int x = (int)current_piece.x;
        int y = (int)current_piece.z;

        return ((y + 4), x + 4);

    }

    

    void getCurrentAvailableMoves()
    {
        List<int> currentPieceMovies = new List<int>();
        foreach(var move in available_moves_current)
        {
            if (move[0] == currentpiece_x && move[1] == currentpiece_y)
            {
                /*currentPieceMovies.Add(move);*/
                GameObject movespot = GameObject.CreatePrimitive(PrimitiveType.Plane);
                movespot.transform.position = new Vector3(move[3] + offsetx, 0.55f, move[2] + offsetz);
                
                movespot.transform.localScale = new Vector3(0.1f, 0.05f, 0.1f);
                moveSpots.Add(movespot);
            }
        }
    }
    
    
    void moveSelectedPiece()
    {
        RaycastHit hit;
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        //print(Input.mousePosition);
        if (Physics.Raycast(ray, out hit, 100.0f))
        {
            if (hit.transform != null)
            {

                GameObject spot;
                if (spot = hit.collider.gameObject)
                {
                    
                    if (spot.name == "Plane")
                    {
                        Debug.Log(available_moves_current.Count);
                        foreach (var movespot in moveSpots)
                        {
                            Destroy(movespot);
                        }
                        /*    Destroy(moveSpotLeft);
                        if (moveSpotRight)
                            Destroy(moveSpotRight);*/
                        int i, j;
                        (i, j) = getBoardCoordinates(spot);
                        (int old_i, int old_j) = getBoardCoordinates(currentPiece);
                        makeAMove(board, old_i, old_j, i, j, "", 7);
                        currentPiece.transform.position = new Vector3(j + offsetx, 0.5f, i + offsetz);
                        
                    }

                }

            }
        }
    }

    public List<List<int>> getInput()
    {
        List<List<int>> available_moves = new List<List<int>>();
        RaycastHit hit;
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            //print(Input.mousePosition);
            if (Physics.Raycast(ray, out hit, 100.0f))
            {
                if (hit.transform != null)
                {

                    GameObject piece;
                    if (piece = hit.collider.gameObject)
                    {
                    int i, j;
                    (i, j) = getBoardCoordinates(piece);
                    (currentpiece_x, currentpiece_y) = getBoardCoordinates(piece);
                    List<int> curPosition = new List<int> { i, j };
                    /*Debug.Log(i);
                    Debug.Log(j);*/

                    
                    if (piece.name == "Cylinder")
                        {

                        available_moves = findPlayerAvailableMoves(board, false);
                        currentPiece = piece;
                        foreach (var coordinates in available_moves)
                        {
                            /*Debug.Log(coordinates[0].ToString() + " " + coordinates[1].ToString());*/
                            /*Debug.Log(coordinates[1]);*/
                            if (coordinates[0] == i && coordinates[1] == j)
                            {
                                /*Debug.Log("Food");*/
                                /*makeAMove(board, i, j, coordinates[2], coordinates[3], "", 7, piece);
                                piece.transform.position = new Vector3(coordinates[3] - 4, 0.5f, coordinates[2] - 4);*/
                                
                            }
                            
                            
                        
                            /*foreach (var y in x)
                                Debug.Log(y.ToString());*/
                        }
                    }

                    }

                }
            }
        return available_moves;
    }

    void aIMakeMove()
    {
        List<List<int>> available_moves = new List<List<int>>();
        available_moves = findAvailableMoves(board, false);

        foreach( var coordinates in available_moves)
        {
            /*Debug.Log(coordinates[2].ToString() + " " + coordinates[3].ToString());*/
        }
        List<int> move = new List<int>();
        if(available_moves.Count != 0) // choosing random move
        {
            move = available_moves[0];
        }

        RaycastHit hit;
        Vector3 camPos = Camera.main.transform.position;
        Ray ray = new Ray(camPos, new Vector3(move[1] + offsetx, 0.5f, move[0] + offsetz) - camPos);
        Debug.Log("Ray: " + (move[1]).ToString() + " " + (move[0]).ToString());
        if (Physics.Raycast(ray, out hit, 100.0f))
        {
            if (hit.transform != null)
            {

                GameObject piece;
                if (piece = hit.collider.gameObject)
                {
                    if (piece.name == "Cylinder")
                    {
                        Debug.Log("Ai piece");
                        int i, j;
                        (i, j) = getBoardCoordinates(piece);
                        Debug.Log(i.ToString() + " " + j.ToString());
                        
                        piece.transform.position = new Vector3(move[3] + offsetx, 0.5f, move[2] + offsetz);

                    }
                    if (piece.name == "Plane")
                    {
                        
                    }
                    
                }

            }
            else
            {
                print("Ray is null");
            }


        }

        Debug.Log(move[0].ToString() + " " + move[1].ToString() + " " + move[2].ToString() + " " + move[3].ToString());
        makeAMove(board, move[0], move[1], move[2], move[3], "", 0);

    }

    bool checkWin()
    {
        if (whitePieces * redPieces == 0)
            return true;
        return false;
    }

    void updatePiecesCount(int x, int y)
    {
        if (getGameObject(x, y).GetComponent<Renderer>().material.color == Color.white)
            whitePieces--;
        else
            redPieces--;

    }

    void boardPositionDebug()
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
    void Start()
    {
        setGame();

        boardPositionDebug();
    }

    List<List<int>> available_moves_current = new List<List<int>>();
    bool playerSelectedPiece = false;
    void Update()
    {
        if(currentState == "PlayerToSelect")
        {
            if (Input.GetMouseButtonDown(0))
            {
                Debug.Log("Mouse Button down");
                available_moves_current = getInput();
                if (available_moves_current != null)
                {
                    Debug.Log("Player has selected piece");
                    getCurrentAvailableMoves();
                }
                
            }
            if (Input.GetMouseButtonUp(0))
            {
                Debug.Log("Mouse Button up");
                currentState = "PlayerToMoveSelectedPiece";
            }
                

        }
        
        if (currentState == "PlayerToMoveSelectedPiece")
        {
            
            if (Input.GetMouseButtonDown(0))
            {
                playerSelectedPiece = true;
                Debug.Log("Mouse Button down");
                moveSelectedPiece();
                Debug.Log("Player has moved selected piece");
            }
            if (Input.GetMouseButtonUp(0) && playerSelectedPiece)
            {
                Debug.Log("Mouse Button up");
                playerSelectedPiece = false;
                currentState = "AiToPlay";

                if (checkWin())
                {
                    currentState = "GameOver";
                    Debug.Log("GameOver");
                }
                    
            }
            

        }
        if (currentState == "AiToPlay")
        {
            aIMakeMove();
            Debug.Log("AI has played");
            currentState = "PlayerToSelect";

            if (checkWin())
            {
                currentState = "GameOver";
                Debug.Log("GameOver");
            }
                
        }

    }
}