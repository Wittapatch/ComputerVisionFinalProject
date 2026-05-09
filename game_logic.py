def decide_winner(player1_gesture, player2_gesture):
    if player1_gesture == "unknown" or player2_gesture == "unknown":
        return "Waiting"
    
    if player1_gesture == player2_gesture:
        return "Draw"
    
    if player1_gesture == "rock" and player2_gesture == "scissors":
        return "Player 1 Wins"
    
    if player1_gesture == "paper" and player2_gesture == "rock":
        return "Player 1 Wins"
    
    if player1_gesture == "scissors" and player2_gesture == "paper":
        return "Player 1 Wins"
    
    return "Player 2 Wins"