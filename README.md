# TOAP
$Elit
let
    Источник = Excel.Workbook(File.Contents("C:\Users\v.nevmerzhytskyi\Downloads\KV20001877 Stock 30-04-18.xlsx"), null, true),
    Items_Sheet = Источник{[Item="Items",Kind="Sheet"]}[Data],
    #"Измененный тип" = Table.TransformColumnTypes(Items_Sheet,{{"Column1", type text}, {"Column2", type text}, {"Column3", type text}, {"Column4", type text}, {"Column5", type text}, {"Column6", type text}, {"Column7", type text}, {"Column8", type text}, {"Column9", type text}, {"Column10", type text}, {"Column11", type text}, {"Column12", type any}}),
    #"Повышенные заголовки" = Table.PromoteHeaders(#"Измененный тип", [PromoteAllScalars=true]),
    #"Измененный тип1" = Table.TransformColumnTypes(#"Повышенные заголовки",{{"Активный код ELIT", type text}, {"Каталожный номер", type text}, {"Бренд", type text}, {"Описание товара", type text}, {"Описание  eCat", type text}, {"Цена клиента", type text}, {"НАЛИЧИЕ", type text}, {"Column8", type text}, {"Column9", type text}, {"Column10", type text}, {"Column11", type text}, {"Column12", type any}}),
    #"Переименованные столбцы" = Table.RenameColumns(#"Измененный тип1",{{"НАЛИЧИЕ", "Ваш филиал, Київ 2"}, {"Column8", "Филиал Київ 3"}, {"Column9", "Филиал Київ 4"}, {"Column10", "Центральный склад"}, {"Column11", "Другие склады, поставка 1-2 дня"}}),
    #"Удаленные столбцы" = Table.RemoveColumns(#"Переименованные столбцы",{"Column12"}),
    #"Удаленные верхние строки" = Table.Skip(#"Удаленные столбцы",1),
    #"Переименованные столбцы1" = Table.RenameColumns(#"Удаленные верхние строки",{{"Каталожный номер", "Артикул"}, {"Цена клиента", "Цена"}}),
    #"Удаленные столбцы1" = Table.RemoveColumns(#"Переименованные столбцы1",{"Описание  eCat"}),
    #"Переименованные столбцы2" = Table.RenameColumns(#"Удаленные столбцы1",{{"Описание товара", "Название"}}),
    #"Удаленные столбцы2" = Table.RemoveColumns(#"Переименованные столбцы2",{"Активный код ELIT"}),
    #"Переупорядоченные столбцы" = Table.ReorderColumns(#"Удаленные столбцы2",{"Бренд", "Артикул", "Цена", "Название", "Ваш филиал, Київ 2", "Филиал Київ 3", "Филиал Київ 4", "Центральный склад", "Другие склады, поставка 1-2 дня"}),
    #"Измененный тип2" = Table.TransformColumnTypes(#"Переупорядоченные столбцы",{{"Ваш филиал, Київ 2", Int64.Type}, {"Филиал Київ 3", Int64.Type}, {"Филиал Київ 4", Int64.Type}, {"Центральный склад", Int64.Type}, {"Другие склады, поставка 1-2 дня", Int64.Type}}),
    #"Добавлен пользовательский объект" = Table.AddColumn(#"Измененный тип2", "Пользовательская", each [#"Ваш филиал, Київ 2"]+[Филиал Київ 3]+[Филиал Київ 4]+[Центральный склад]+[#"Другие склады, поставка 1-2 дня"]),
    #"Переименованные столбцы3" = Table.RenameColumns(#"Добавлен пользовательский объект",{{"Пользовательская", "Кол-во"}}),
    #"Условный столбец добавлен" = Table.AddColumn(#"Переименованные столбцы3", "Пользовательская", each if [#"Ваш филиал, Київ 2"] >= 1 then 1 else if [Филиал Київ 3] >= 1 then 1 else if [Филиал Київ 4] >= 1 then 1 else if [Центральный склад] >= 1 then 2 else if [#"Другие склады, поставка 1-2 дня"] >= 1 then 3 else null),
    #"Переименованные столбцы4" = Table.RenameColumns(#"Условный столбец добавлен",{{"Пользовательская", "Срок поставки"}}),
    #"Удаленные столбцы3" = Table.RemoveColumns(#"Переименованные столбцы4",{"Ваш филиал, Київ 2", "Филиал Київ 3", "Филиал Київ 4", "Центральный склад", "Другие склады, поставка 1-2 дня"}),
    #"Переупорядоченные столбцы1" = Table.ReorderColumns(#"Удаленные столбцы3",{"Бренд", "Артикул", "Цена", "Кол-во", "Срок поставки", "Название"}),
    #"Строки с примененным фильтром" = Table.SelectRows(#"Переупорядоченные столбцы1", each [#"Кол-во"] >= 2)
in
    #"Строки с примененным фильтром"
