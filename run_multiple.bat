@echo off
for /l %%i in (1, 1, 5) do (
	start "c%%i" docker run --rm --name=fl-server-container%%i -v "D:\BUET\2023-04\Distributed Systems\Project\Codes\ProjectTest\dataset":/app/dataset fl-server
)