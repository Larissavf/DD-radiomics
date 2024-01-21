#imports
import numpy as np
import pandas as pd
import panel as pn
import bokeh as bk
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.models import HoverTool, ColumnDataSource, Whisker
from io import StringIO

pn.extension()



#main dataset
temp_df = pd.DataFrame()

#options for group choises
df_radionmics = []
options = []
options_group = []

#Functions

def create_scater_plot(z, title,  df_clini, df_radionmics):
    if z == "options":
        return
    #grab group column
    group = df_clini.loc[:,z]
    #unique factors
    uniq = group.unique()

    if z not in df_radionmics: 
        #paste to the frame with the data for the plot
        group = [str(item) for item in group]
        df_radionmics.insert(0, z, group)
    #order data ascending
    df = df_radionmics.sort_values(by=[z], ascending=True)

    # Create figure object
    TOOLS = "reset,tap,save"
    p = figure(title = title, tools=TOOLS)

    #create colormap for different lengths
    if len(uniq) == 1:
        color = "#1f77b4"
    elif len(uniq) <= 2:
        uniq = [str(item) for item in uniq]
        color = factor_cmap(z, ["#1f77b4", "#aec7e8"], uniq)
    elif len(uniq) <= 20:
        uniq = [str(item) for item in uniq]
        color = factor_cmap(z, bk.palettes.Category20[len(uniq)], uniq)
    elif len(uniq) > 20:
        # create how big the bins will be if you take 20 colors
        max_value = int(uniq.max())
        min_value = int(uniq.min())
        step = max(1, 3*int(round((max_value - min_value)/10, 0)))
        bins = list(range(min_value, max_value + step, step))
        # create the groups and append to dataframe
        new_z = []
        count=0
        for x in group:
            count +=1
            count_value = 0  
            for value in bins[:-1]:
                if bins[count_value] >= float(x) and float(x) < bins[count_value + 1]:
                    new_z.append(bins[count_value])
                    break  
                count_value += 1
        
        # add to datafram by removing the old first
        df.drop([z], axis=1 ,inplace=True)
        df.insert(0, z, new_z)
        df = df.sort_values(by=[z], ascending=True)
        #make colormap
        color = linear_cmap(z, bk.palettes.Category20[10], low=min(bins), high=max(bins))

    return p, color, df

def create_plot_heat( x, all, title, df_radionmics):
    #grab the correct collums according to the widget
    temp_df = {}
    for element in x:
        temp = df_radionmics[element]
        temp_df[element] = temp
    df = pd.DataFrame(data = temp_df)
 
    if all: 
        df = df_radionmics
    
    # Create the correlation df
    df_hm = df.corr()

    # Make the plot
    p = df_hm.hvplot.heatmap(title= title, 
                    colorbar=False, 
                    cmap="fire_r", 
                    xlabel="Feature 1",
                    ylabel= "Feature 2",
                    height = 700,
                    width = 700)
    p.opts(xrotation = 45,
        tools=["hover"])

    return p

def create_plot_f1(x, z, title,  df_clini, df_radionmics):
    #create figure and prep df and colorpallete 
    p, color, df = create_scater_plot(z, title,  df_clini, df_radionmics)

    p.scatter("patient", x, source=df, color=color, legend_field=z)
    hover = HoverTool(tooltips=[("Patient ID", "@patient"),
    ("y-waarde","$x")])
    p.add_tools(hover)
    return p

def create_plot_f2(x, y, z, title, df_clini, df_radionmics):
    #create figure and prep df and colorpallete 
    p, color, df = create_scater_plot(z, title,  df_clini, df_radionmics)
    p.scatter(x, y, source=df, color=color, legend_field=z)
    hover = HoverTool(tooltips=[("Patient ID", "@patient"),
    ("y-waarde","$x"),
    ("x-waarde","$x")])
    p.add_tools(hover)
    return p

def create_boxplot(df_shape, features):
    #grab the correct collums according to the widget
    temp_df = {}
    for element in features:
        temp = df_shape[element]
        temp_df[element] = temp
    df_shape = pd.DataFrame(data = temp_df)
    #unique keys
    kinds = list(df_shape.columns)

    #create quantile
    qs = df_shape.quantile([0.25, 0.5, 0.75])
    qs.insert(0,"new_index", ["q1", "q2", "q3"])
    temp_qs = qs.set_index("new_index")


    #create lower and upper
    upper=["upper"]
    lower=["lower"]
    for column in temp_qs:
        q1 = temp_qs.loc["q1",column]
        q3 = temp_qs.loc["q3",column]
        iqr = q3 - q1
        upper.append(q3 + 1.5*iqr)
        lower.append(q1 - 1.5*iqr)
    df_shape.reset_index(inplace=True)
    melt = pd.melt(df_shape, id_vars=['patient'], value_vars=df_shape.columns)


    #add to df
    qs.loc[len(qs.index)] = upper
    qs.loc[len(qs.index)] = lower
    qs.set_index("new_index", inplace=True)

    #switch columns with index
    New_df=qs.T.groupby(level=0).agg(lambda x : x.values.tolist()).stack().apply(pd.Series).unstack().sort_index(level=1,axis=1)
    New_df.columns=New_df.columns.droplevel(level=0)
    New_df.reset_index(inplace=True)

    #merge the quantile etc with the data
    df_boxplot = pd.merge(melt, New_df, left_on="variable", right_on="index", how="right").drop("index", axis=1)

    source = ColumnDataSource(df_boxplot)

    p = figure(x_range=kinds, tools="hover",
            title="Distribution with each feature",
            background_fill_color="#eaefef", y_axis_label="The unit of the feature")

    # outlier range
    whisker = Whisker(base="variable", upper="upper", lower="lower", source=source)
    whisker.upper_head.size = whisker.lower_head.size = 20
    p.add_layout(whisker)

    # quantile boxes
    cmap = factor_cmap("variable", "TolRainbow7", kinds)
    p.vbar("variable", 0.7, "q2", "q3", source=source, color=cmap, line_color="black")
    p.vbar("variable", 0.7, "q1", "q2", source=source, color=cmap, line_color="black")

    # outliers
    outliers = df_boxplot[~df_boxplot.value.between(df_boxplot.lower, df_boxplot.upper)]
    p.scatter("variable", "value", source=outliers, size=6, color="black", alpha=0.3)

    p.xgrid.grid_line_color = None
    p.axis.major_label_text_font_size="14px"
    p.axis.axis_label_text_font_size="12px"
    p.axis.major_label_orientation = 45

    return p


def add_graph_to_layout(items, event):
    add_graphbox = items[0]
    select_graphtype = items[1]
    graph_layout = items[2]
    if not event:
        return
    # make invisible
    if add_graphbox.visible:
        add_graphbox.visible = False
    else:
        #make visibele
        select_graphtype.value = "Options"
        select_graphtype.disabled_options = reset_disabledoptions(graph_layout)
        add_graphbox.visible = True

def reset_disabledoptions(graph_layout):
    disable = []
    for graph in graph_layout:
        if graph == "scatter with 1 feature":
            disable.append("Scatter(1f)")
        elif graph == "scatter with 2 features":
            disable.append("Scatter(2f)")
        elif graph == "Heatmap":
            disable.append("Heatmap")
        elif graph == "Boxplot":
            disable.append("Boxplot")
    return disable
            


def graphtype_disabledoptions(select_graphtype):
    #grab the already disabled graph types
    disablegraphs = []
    alreadydisabled = select_graphtype.disabled_options
    for element in alreadydisabled:
        disablegraphs.append(element)

    #add the new one
    value = select_graphtype.value
    select_graphtype.value = "Options"
    if value != "Options":
        disablegraphs.append(value)
    
    # set value
    select_graphtype.disabled_options = disablegraphs
    return

def show_graphtype_in_body(items , event):
    #unpack variables
    add_graphbox = items[0]
    select_graphtype = items[1]
    dashboard_body = items[2]
    graph_layout = items[3]

    if not event:
        return
    
    #add the selected graph type to body
    graphtype = event.new
    if graphtype == "Options":
        return 
    elif graphtype == "Scatter(1f)":
        #to keep the overview of the plots
        graph_layout.append("scatter with 1 feature")
        # the new body
        scatter1f = pn.Row(pn.layout.Divider(),
        pn.Column("### Create your scatter plot with 1 value", 
                pn.WidgetBox(input_title, autoinput_f1, select_group, height = 500),
                btn_delete_f1
                ),
        pn.Column(fig_f1)
        )
        # append to body
        dashboard_body.append(scatter1f)
        dashboard_body.object = dashboard_body 
    elif graphtype == "Heatmap":
        #to keep the overview of the plots
        graph_layout.append("Heatmap")
        # the new body
        heatmap = pn.Row(pn.layout.Divider(),
        pn.Column("### Create your heatmap of the correlation between the features", 
                pn.WidgetBox(input_title_hm, multi_choice, checkbox_hm, height = 500),
                btn_delete_hm   
                ),
        pn.Column(fig_hm)
        )
        # append to body
        dashboard_body.append(heatmap)
        dashboard_body.object = dashboard_body 
    elif graphtype == "Scatter(2f)":
        #to keep the overview of the plots
        graph_layout.append("scatter with 2 features")
        # the body
        scatter2f = pn.Row(pn.layout.Divider(),
            pn.Column( "### Create your scatter plot with 2 values",
                      pn.WidgetBox(input_title_f2, autoinput_f2_x, autoinput_f2_y, select_group_f2, height = 500), 
                      btn_delete_f2
                      ),
            pn.Column(fig_f2)
        )
        #append to the body
        dashboard_body.append(scatter2f)
        dashboard_body.object = dashboard_body 
    elif graphtype == "Boxplot":
        #to keep the overview of the plots
        graph_layout.append("Boxplot")
        # the new body
        boxplot = pn.Row(pn.layout.Divider(),
            pn.Column("### Create your boxplots of the distribution from the features", 
                    pn.WidgetBox(multi_choice_bp, height = 500),
                    btn_delete_bp 
                    ),
            pn.Column(fig_bp)
            )
        # append to body
        dashboard_body.append(boxplot)
        dashboard_body.object = dashboard_body 
        return

    #make invisible
    add_graphbox.visible = False
    graphtype_disabledoptions(select_graphtype)
        
    return

def read_inputfiles(file, radionmics = False):
    # transform the file to a df
    df = pd.read_csv(StringIO(file.decode("utf-8")))
    if radionmics:
        df.pop("Unnamed: 0")
    df.set_index("patient", inplace=True)

    return df


def file_radionomics(df_widget, event):
    file = event.new
    if file != event.old:
        if file:
            df = read_inputfiles(file, radionmics = True)

            # edit the widget
            df_widget.value = df
            df_widget.visible = True
            return
    return 

def file_clinical(df_widget, event):
    file = event.new
    if file != event.old:
        if file:
            df = read_inputfiles(file)

            # edit the widget
            df_widget.value = df
            df_widget.visible = True
            return 
    return 

def delete_graph(items, event):
    dashboard_body = items[0]
    graph_layout = items[1]

    locations = dict(enumerate(graph_layout))
    button = event.obj.description[26:]
    for loc, graphtype in locations.items():
        if graphtype == button:
            dashboard_body.pop(loc)
            graph_layout.remove(graphtype)

def select_features(dataframe_radio, event):
    #standard feature values
    feature_indices = {"shape features": 23, "first order features": 38, "GLCM features": 55,
                         "GLDM features": 79, "GLRLM features": 93, "GLZSM_features": 109, "NGTDM_features": 125}
    indices = [23 ,38 ,55, 79, 93, 109, 125]

    features = event.new
    df = dataframe_radio[0].value
    df_old = pd.DataFrame()

    btn = dataframe_radio[2] 

    for feature in features:
        indx = feature_indices[feature]
        if indx == 125:
            if not df_old.empty:
                df_new = df.iloc[:,indx:]
                df_old = pd.concat([df_old, df_new], axis=1 )
            else:
                df_old = df.iloc[:,indx:]
        else:
            if not df_old.empty:
                numb = [i for i,x in enumerate(indices) if x == indx]
                df_new = df.iloc[:, indx: indices[numb[0]+ 1] -1]
                df_old = pd.concat([df_old, df_new], axis=1)
            else:
                #aangegeven feature tot the volgende 
                numb = [i for i,x in enumerate(indices) if x == indx]
                df_old = df.iloc[:, indx: indices[numb[0]+ 1] -1]
    
    #change the col and index names
    ticks_names = list(df_old.columns)
    ticksnames = {}
    for names in ticks_names:
        ticksnames[names] = names.replace("_", " ")
    df_old.rename(index = ticksnames, columns= ticksnames, inplace=True)

    #make new dataframe visible
    dataframe_radio[1].value = df_old
    dataframe_radio[1].visible = True

    #make button visible
    btn.visible = True
    
    return df_old

def change_options(items, event):
    autoinput_f1 = items[0]
    autoinput_f2_x = items[1] 
    autoinput_f2_y = items[2]
    multi_choice = items[3]
    multi_choice_bp = items[4]

    dataframe = event.new
    options = list(dataframe.columns)

    autoinput_f1.options = options
    autoinput_f2_x.options = options
    autoinput_f2_y.options = options
    multi_choice.options = options

    multi_choice.value = [options[0]]
    multi_choice_bp.options = options
    multi_choice_bp.value = [options[0]]

    return

def change_options_group(items, event):
    select_group = items[0]
    select_group_f2 = items[1] 

    dataframe = event.new
    df_nonan = dataframe.dropna(axis=1)
    select_group.options = list(df_nonan.columns)
    select_group_f2.options = list(df_nonan.columns)


dashboard = pn.template.BootstrapTemplate(
    title= "PlotRadionomics",
    header_background = "indigo",
    sidebar=[]
)

#Create widgets
## main
btn_add_graph = pn.widgets.Button(
    name="Add graph", 
    icon="plus", 
    button_type="primary", 
    description="Push to add a another graph", 
    visible=False
    )
select_graphtype = pn.widgets.Select(
    name="Choose your graph type", 
    options=["Options", "Scatter(1f)", "Scatter(2f)", "Boxplot", "Heatmap"], 
    disabled_options=[], 
    value="Options"
    )
radionomics_input = pn.widgets.FileInput(accept=".csv")
clinical_input = pn.widgets.FileInput(accept=".csv")
multi_choice_df = pn.widgets.MultiChoice(name='Choose the features you want to compare',
    options=["shape features", "first order features", "GLCM features",
              "GLDM features", "GLRLM features", "GLZSM features", "NGTDM features"])
pn_df_clinical = pn.widgets.DataFrame(temp_df ,visible = False,  reorderable= True )
pn_df_radionmics = pn.widgets.DataFrame(temp_df ,visible = False, reorderable= True)
pn_df_edit = pn.widgets.DataFrame(temp_df, visible= False, reorderable= True)


## Scatter(1f)
autoinput_f1 = pn.widgets.AutocompleteInput(
    name="Filter on feature", 
    options=options,
    case_sensitive=False, 
    search_strategy="includes",
    placeholder="Start to write feature"
    )
select_group = pn.widgets.Select(
    name="Based on wich group", 
    options=options_group,
    value = "Sex")
input_title = pn.widgets.TextInput(
    name="Graph title",
      placeholder="Enter a string here..."
      )
btn_delete_f1 = pn.widgets.Button(
    name="delete", 
    button_type="danger", 
    description="Push to delete the graph: scatter with 1 feature", 
)

## Heatmap
input_title_hm = pn.widgets.TextInput(
    name="Graph title",
      placeholder="Enter a string here..."
      )
multi_choice = pn.widgets.MultiChoice(name='Choose the features you want to compare',
    options=options,
    value = ["original shape Flatness"])
checkbox_hm = pn.widgets.Checkbox(name='All features')
btn_delete_hm = pn.widgets.Button(
    name="delete", 
    button_type="danger", 
    description="Push to delete the graph: Heatmap", 
)


## Scatter(2f)
autoinput_f2_x = pn.widgets.AutocompleteInput(
    name="Filter on feature for x-axis", 
    options=options,
    case_sensitive=False, 
    search_strategy="includes",
    placeholder="Start to write feature"
    )
autoinput_f2_y = pn.widgets.AutocompleteInput(
    name="Filter on feature for y-axis", 
    options=options,
    case_sensitive=False, 
    search_strategy="includes",
    placeholder="Start to write feature"
    )
select_group_f2 = pn.widgets.Select(
    name="Based on wich group", 
    options=options_group,
    value = "Sex"
    )
input_title_f2 = pn.widgets.TextInput(
    name="Graph title",
      placeholder="Enter a string here..."
      )
btn_delete_f2 = pn.widgets.Button(
    name="delete", 
    button_type="danger", 
    description="Push to delete the graph: scatter with 2 features", 
)

## boxplot
multi_choice_bp = pn.widgets.MultiChoice(name='Choose the features you want to see',
    options=options,
    value = ["original shape Flatness"])
btn_delete_bp = pn.widgets.Button(
    name="delete", 
    button_type="danger", 
    description="Push to delete the graph: Boxplot", 
)


#Bind widgets

## binding of the different figures with the widgets
fig_f1 = pn.bind(create_plot_f1, x=autoinput_f1, z=select_group, title=input_title, df_clini=pn_df_clinical, df_radionmics=pn_df_edit)
fig_f2 = pn.bind(create_plot_f2, x=autoinput_f2_x, y=autoinput_f2_y, z=select_group_f2, title=input_title,  df_clini=pn_df_clinical, df_radionmics=pn_df_edit)
fig_hm = pn.bind(create_plot_heat, title= input_title_hm, x=multi_choice, all=checkbox_hm,  df_radionmics=pn_df_edit)
fig_bp = pn.bind(create_boxplot,  df_shape=pn_df_edit, features=multi_choice_bp)

#Create body items
## contains the menu of what graph you want to select
add_graphbox = pn.WidgetBox(
    "## Create your graph", 
    select_graphtype, visible = False
    )

## welcome element
welcome = pn.Column(
    "# Welcome at PlotRadionomics",
    """Here you can make different plots of your radionomics data.
    For this you also need a clinical file with data of all the same patients in your radionomics.
    This data can be info you can filter the dataset on.
    Only csv files are accepted.""",
    pn.Row("Upload your file with the radionomics data here", radionomics_input,
    "Upload your file with the clinical patient data here", clinical_input),
    pn.Row(pn_df_radionmics, pn_df_clinical),
    pn.Row(multi_choice_df),
    pn.Row(pn_df_edit))

#the body
body = pn.Column(pn.Column(
            welcome,
            "## Want to add a graph?", 
            "First upload the files and chose your features",
            btn_add_graph, add_graphbox
            ), css_classes=['flex-container'])

#list with all the graphtypes in the body
graph_layout = ["welcome"]

#watch the widgets
## add graph to the body
items = [add_graphbox, select_graphtype, body, graph_layout]
select_graphtype.link(items, callbacks={"value": show_graphtype_in_body})
## takes care of the graph select menu
items = [add_graphbox, select_graphtype, graph_layout]
btn_add_graph.link(items, callbacks={"value": add_graph_to_layout})
## takes care of the feature select menu
items = [pn_df_radionmics, pn_df_edit, btn_add_graph]
multi_choice_df.link(items, callbacks={"value": select_features})
## delete graphs of layout
items = [body, graph_layout]
btn_delete_f1.link(items, callbacks={"value": delete_graph})
btn_delete_f2.link(items, callbacks={"value": delete_graph})
btn_delete_hm.link(items, callbacks={"value": delete_graph})
## binding of the dataframes to the page
radionomics_input.link(pn_df_radionmics, callbacks={"value": file_radionomics})
clinical_input.link(pn_df_clinical, callbacks={"value": file_clinical})
## edit options in widgets
items = [select_group, select_group_f2]
pn_df_clinical.link(items, {"value": change_options_group})
items = [autoinput_f1, autoinput_f2_x, autoinput_f2_y, multi_choice, multi_choice_bp]
pn_df_edit.link(items, {"value": change_options})



dashboard.main.append(body)


dashboard.servable()


stylesheet = """
flex-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;  
}

"""