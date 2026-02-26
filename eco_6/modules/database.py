"""
Uses the minimal APSW to interface with sqlite3
One object per table/db access

see /examples/
"""
import sqlite3

class Table:
    def __init__(self, file: str, table: str):
        """
        Create a new Database-Table instance\n
        ex. file="db/file.db"\n
        table="main_a"\n
        Run with intended filepath and name to create .db
        """
        self.connect = sqlite3.connect(file)
        self.table = table
        self.cursor = self.connect.cursor()
    
    def executeSQL(self, rawQuery: str, funcName: str = "<unspecified>", debug: bool = False):
        """
        Execute raw SQL\n
        Use "{0}" for table if using outside of class\n
        Main function for class, error handling\n
        Set debug=True to print the raw query
        """
        try:
            r = rawQuery.format(self.table) # insert table name in place of "{0}"
            e = self.cursor.execute(r)
            if debug:
                print(r)
                #print(dir(e))
            return e
        
        except sqlite3.OperationalError as err:
            print(r)
            print(f"! operational err !\neco.db.Table\n\t.{funcName} # might be missing ''\n\t.executeSQL():\n\t\t{err}\n")
        
        except sqlite3.IntegrityError as err:
            print(r)
            print(f"! integrity err !\neco.db.Table\n\t.{funcName} # might be missing ''\n\t.executeSQL():\n\t\t{err}\n")
    
    
    # --------------------------- utils ---------------------------
    def refresh():
        """
        Close and reopen database connection\n
        Good for verifying a write op\n
        not implemented yet
        """
        pass

    def begin(self):
        """Start a block write"""
        self.sql("BEGIN TRANSACTION", "begin()")
    
    def commit(self):
        """Finish a block write to write all at once"""
        self.sql("COMMIT", "commit()")
    
    def vacuum(self):
        """
        Cleanup existing indices\n
        Can help with lowering filesize after deletions
        """
        self.sql("VACUUM", "vacuum()")

    # aliases
    sql = executeSQL
    start = begin
    finish = commit
    end = commit
    cleanup = vacuum


    # --------------------------- table ---------------------------
    def createNewTable(self, newColArr: list):
        """
        Create a new table from scratch\n
        newColArr = [\n
        \t"id INTEGER PRIMARY KEY",\n
        \t"unix INTEGER UNIQUE",\n
        \t"description TEXT",\n
        \t"float_val REAL"
        ]
        """
        
        joinedCol = ", ".join(newColArr) # join first
        self.sql(
            f"CREATE TABLE {self.table}({joinedCol})", 
            f"createNewTable({newColArr})"
        )

    def createOneColInTable(self, colToAdd: str):
        """
        Create a new column within a table\n
        colToAdd="name TYPE"\n
        ex. colToAdd="description TEXT"\n
        ex. colToAdd="pid INTEGER UNIQUE" etc
        """
        
        self.sql(
            f"ALTER TABLE {self.table} ADD COLUMN {colToAdd}",
            f"createOneColInTable({colToAdd})"
        )
    
    def deleteColsInTable(self, colToDrop:str):
        """Delete/drop columns within a table"""
        self.sql(
            f"ALTER TABLE {self.table} DROP COLUMN {colToDrop}",
            f"deleteColsInTable({colToDrop})"
        )

    # --------------------------- row ---------------------------
    def createRow(self, colNameArr: list, valueArr: list):
        """
        Create new data rows\n
        ex. colNameArr=[title, score], valueArr=["Goldeneye", 85.5]\n
        Remember to use commit() after all ops to commit to db
        """
        
        # wrap originally string-types in 'single quotes'
        # then convert everything to string, so [13, "abcd"] will become ["13", "'abcd'"]
        valStrArr = [(f"\'{str(item)}\'") if type(item) == str else str(item) for item in valueArr]
        
        # join
        nameJoined = ", ".join(colNameArr)
        valJoined = ", ".join(valStrArr)
        self.sql(
            f"INSERT INTO {self.table} ({nameJoined}) VALUES({valJoined})",
            f"createRow(nameJoined={nameJoined}, valJoined={valJoined})", 
            debug=False
        )

    def readAsync(self, selectCols: str = "*", where: str = "true") -> list:
        """
        Read rows chaotically\n
        Try to specify selectCols so you know what the order is\n
        ex. selectCols="unix, pid, a_0, a_1"
        """
        
        res = self.sql(
            f"SELECT {selectCols} FROM {self.table} WHERE {where}",
            f"readAsync(selectCols={selectCols}, where={where})",
            debug=False
        )
        
        return res.fetchall() # other methods like .rowcount aren't correct?

    def readAndOrder(self, orderCol: str, selectCols: str = "*", where: str = "true") -> list:
        """
        Read rows in a particular order\n
        ex. orderCol="unix ASC"\n
        ex. orderCol="pid DESC" etc
        """
        res = self.sql(
            f"SELECT {selectCols} FROM {self.table} WHERE {where}",
            f"readAndOrder(orderCol={orderCol}, selectCols={selectCols}, where={where})",
            debug=False
        )
        
        return res.fetchall()

    def updateOne(self, setCol: str, toValue: str|int|float|bool, where: str = "true"):
        """
        Update one col where rows match <condition>, use with start/finish blocks\n
        ex. setCol="score", toValue=75.2, where="pid = 1"
        """
        
        # wrap originally string-types in 'single quotes'
        # then convert everything to string, so 13 => "13" and "abcd" => "'abcd'"
        valStr = (f"\'{str(toValue)}\'") if type(toValue) == str else str(toValue)
        
        self.sql(
            f"UPDATE {self.table} SET {setCol} = {valStr} WHERE {where}",
            f"updateOne(setCol={setCol}, toValue={valStr}, where={where})",
            debug=False
        )

    def updateMany(self, setColArr: list, toValArr: list, where: str = "true"):
        """
        Update many cols where rows match <condition>, use with start/finish blocks\n
        ex. setColArr=[]
        """
        
        # wrap originally string-types in 'single quotes'
        # then convert everything to string, so [13, "abcd"] will become ["13", "'abcd'"]
        valStrArr = [(f"\'{str(item)}\'") if type(item) == str else str(item) for item in toValArr]
        
        zipArr = [tuple(setColArr), tuple(valStrArr)]
        zipArr = list(zip(*zipArr))
        
        # go thru and string concat col_a = val_a, col_b = val_b, ... etc
        concatArr = [f"{item[0]} = {item[1]}" for item in zipArr]
        
        catJoined = ", ".join(concatArr)
        self.sql(
            f"UPDATE {self.table} SET {catJoined} WHERE {where}",
            f"updateMany(setCol={setColArr}, toValue={toValArr}, where={where})",
            debug=False
        )
    
    def update(self, setCol: list|str, toVal: any, where: str = "true"):
        """Helper function for quick update"""
        if(isinstance(setCol, list) and isinstance(toVal, list)):
            # both are lists, send to many
            self.updateMany(setCol, toVal, where)
        elif((not isinstance(setCol, list)) and (not isinstance(toVal, list))):
            # both not a list, string or number, send to one
            self.updateOne(setCol, toVal, where)
        else:
            print(f"eco.db.update() ran into an unexpected condition!")

    def deleteWhere(self, where: str = "false"):
        """
        Delete many rows, use with start/finish blocks\n
        defaults to where="false" due to its destructive behavior
        """
        self.sql(
            f"DELETE FROM {self.table} WHERE {where}",
            f"deleteWhere({where})"
        )
    
    
    # --------------------------- stats ---------------------------
    def totalRows(self, colToCount: str = "*", where: str = "true"):
        """
        Get total number of rows matching where <condition>\n
        colToCount defaults to * (all) because counting happens on there where <condition
        """
        
        res = self.readAsync(f"COUNT({colToCount})", where=where)
        return res[0][0] # sqlite does a funny thing with count, max, and min, so get the int val
    
    def maxEntryOnCol(self, colToFindMax: str, where: str = "true"):
        """
        Get largest numerical entry matching where <condition>
        """
        res = self.readAsync(f"MAX({colToFindMax})", where=where)
        return res[0][0]
    
    def minEntryOnCol(self, colToFindMin: str, where: str = "true"):
        """
        Get smallest numerical entry matching where <condition>
        """
        res = self.readAsync(f"MIN({colToFindMin})", where=where)
        return res[0][0]
    
    
    # --------------------------- future ---------------------------
    """
    textSearch that uses glob or like
    groupBy
    
    avg and sum are already deprecated, as any real math should be done on gpu
    
    sqlite is only for read/write
    """