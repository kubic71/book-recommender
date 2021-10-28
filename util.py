def unescape(path, dest, enc):
    import html
    with open(path, "r", encoding="iso-8859-1") as f:
        g = open(dest, "w", encoding="utf-8")
        g.write(html.unescape(f.read()))
        g.close()

unescape("BX-Books.csv", "books.csv", enc="iso-8859-1")
unescape("BX-Users.csv", "users.csv", enc="iso-8859-i")
unescape("BX-Book-Ratings.csv", "ratings.csv", enc=None)
