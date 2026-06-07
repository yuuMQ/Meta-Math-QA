from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('sail/Sailor-4B-Chat')
text = "[SYS]Bạn là trợ lý toán học.[/SYS][USR]1+1=?[/USR][AST]"
ids = tokenizer.encode(text)
print(tokenizer.decode(ids))