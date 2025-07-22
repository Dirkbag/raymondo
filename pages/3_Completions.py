import streamlit as st 
import pandas as pd
from supabase import create_client
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from io import BytesIO
import xlsxwriter

# Page setup
st.set_page_config(
    page_title="📋 Completions",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Completions")

# Supabase connection
supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_SERVICE_KEY"]
)

# Load data
data = supabase.table("completions").select("*").execute()
df = pd.DataFrame(data.data)

if df.empty:
    st.info("No completions found.")
    st.stop()

# Configure AgGrid
gb = GridOptionsBuilder.from_dataframe(df)

# Lock ID column
gb.configure_column("id", editable=False, hide=False)

# Column formatting
if "uploaded_at" in df.columns:
    gb.configure_column("uploaded_at", type=["dateColumnFilter","customDateTimeFormat"], custom_format_string='yyyy-MM-dd HH:mm:ss', pivot=True)
if "filesize_kb" in df.columns:
    gb.configure_column("filesize_kb", type=["numericColumn"], precision=2)

# Enable grouping, filtering, pivoting
gb.configure_default_column(
    editable=True,
    filter=True,
    sortable=True,
    resizable=True,
    enablePivot=True,
    enableRowGroup=True,
    enableValue=True,
    minWidth=150
)

# Enable sidebar with full pivot panel support
gb.configure_side_bar()

# Remove pagination to enable vertical scrolling
# (Pagination and vertical scroll conflict in AgGrid)
# gb.configure_pagination(enabled=True, paginationPageSize=10)

gb.configure_selection("multiple", use_checkbox=True)

grid_options = gb.build()

# Inject CSS to force scroll
st.markdown(
    """
    <style>
    .ag-root-wrapper {
        max-height: 500px !important;
        overflow-y: auto !important;
    }
    .ag-body-viewport {
        max-height: 400px !important;
        overflow-y: scroll !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Render AgGrid
preferred_height = 700

grid_response = AgGrid(
    df,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    allow_unsafe_jscode=True,
    enable_enterprise_modules=True,
    fit_columns_on_grid_load=False,
    use_container_width=False,
    height=preferred_height,
    domLayout="normal",
    theme="streamlit"
)

if grid_response is None:
    st.error("❌ AgGrid returned None. The grid may have failed to render.")
    st.stop()

edited_df = grid_response.get("data", df)
selected_rows = grid_response.get("selected_rows", [])

# Save edits
if st.button("💾 Save changes"):
    for _, row in edited_df.iterrows():
        if "id" in row and row["id"]:
            supabase.table("completions").update(row.to_dict()).eq("id", row["id"]).execute()
    st.success("✅ All changes saved to Supabase.")

# Row deletion
if selected_rows and len(selected_rows) > 0:
    with st.expander("🗑 Delete Selected Rows"):
        st.warning(f"Delete {len(selected_rows)} selected row(s)?")
        if st.button("Confirm Delete"):
            for row in selected_rows:
                if row.get("id"):
                    supabase.table("completions").delete().eq("id", row["id"]).execute()
            st.success("Rows deleted. Please refresh to see changes.")

# Export options
st.markdown("### 📤 Export Data")
export_format = st.selectbox("Select export format", ["CSV", "Excel"])

if st.button("Download Data"):
    if export_format == "CSV":
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="completions.csv", mime="text/csv")
    else:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Completions')
            workbook = writer.book
            worksheet = writer.sheets['Completions']
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'align': 'center',
                'bg_color': '#D7E4BC',
                'border': 1
            })
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                worksheet.set_column(col_num, col_num, 20)
        st.download_button(
            label="📥 Download Excel",
            data=output.getvalue(),
            file_name="completions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
