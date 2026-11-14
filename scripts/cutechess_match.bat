@echo off
REM Compares the engine WITH the NNUE net against the SAME engine WITHOUT it (so only the net differs).
REM Needs cutechess-cli, an opening book (.pgn or .epd) and the engine exe. Edit the paths below.
cd /d "%~dp0\.."
cutechess-cli.exe ^
-engine name="nnue" cmd="%CD%\chess-ai.exe" proto=uci option.EvalFile="%CD%\nets\net.bin" option.BookFile=none option.MoveOverhead=30 option.Hash=64 option.Threads=1 ^
-engine name="hce"  cmd="%CD%\chess-ai.exe" proto=uci option.BookFile=none option.MoveOverhead=30 option.Hash=64 option.Threads=1 ^
-each tc=5+0.05 ^
-rounds 5000 -games 2 -repeat ^
-concurrency 3 ^
-openings file="%CD%\books\openings.pgn" format=pgn order=random ^
-sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 ^
-ratinginterval 20 ^
-pgnout "%CD%\results\nnue_vs_hce.pgn" ^
-recover
pause
