import streamlit as st
import pandas as pd
from supabase import create_client
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from io import BytesIO
import xlsxwriter

# --- 1. Page Setup ---
st.set_page_config(
    page_title="📋 Document Management",
    page_icon="�",
    layout="wide"
)
st.title("📋 Document Management & Diagnostics")
st.markdown("Manage uploaded documents and verify they have been processed into chunks for the chatbot.")

# --- 2. Supabase Connection & Data Fetching ---
@st.cache_data(ttl=600) # Cache data for 10 minutes
def load_data(_supabase_client):
    """
    Loads data from both tables and calculates chunk counts.
    The _supabase_client parameter has a leading underscore to prevent Streamlit from trying to hash it.
    """
    # Get the list of uploaded documents, ordered by ID to show newest first.
    uploaded_docs_response = _supabase_client.table("uploaded_documents").select("*").order("id", desc=True).execute()
    df = pd.DataFrame(uploaded_docs_response.data)

    if df.empty:
        return df

    # --- Calculate chunk count for each document ---
    st.info("Calculating chunk counts for each document... This may take a moment.")
    progress_bar = st.progress(0)
    df['chunk_count'] = 0 # Initialize column

    for i, row in df.iterrows():
        file_name = row['name']
        try:
            # Query the 'documents' table to count chunks matching the source filename
            count_response = _supabase_client.table("documents").select('id', count='exact').eq('metadata->>source', file_name).execute()
            chunk_count = count_response.count
            df.loc[i, 'chunk_count'] = chunk_count
        except Exception as e:
            st.error(f"Could not get chunk count for {file_name}: {e}")
            df.loc[i, 'chunk_count'] = -1 # Indicate an error
        
        progress_bar.progress((i + 1) / len(df))
    
    progress_bar.empty()
    return df

try:
    # Initialize the Supabase client in the main scope
    supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_SERVICE_KEY"])
    
    # Pass the client to the data loading function
    df = load_data(supabase)

    if df.empty:
        st.warning("No uploaded documents found. Use the 'Document Uploader' page to add some.")
        st.stop()

    # --- 3. AgGrid Configuration ---
    gb = GridOptionsBuilder.from_dataframe(df)

    # Configure the new chunk_count column
    gb.configure_column("chunk_count", header_name="Chunk Count", type=["numericColumn"], width=150, editable=False)
    gb.configure_column("id", editable=False, hide=False, width=100)
    gb.configure_column("name", header_name="File Name", width=400)

    if "created_at" in df.columns:
        gb.configure_column("created_at", header_name="Upload Date", type=["dateColumnFilter","customDateTimeFormat"], custom_format_string='yyyy-MM-dd HH:mm', width=200)

    gb.configure_default_column(editable=True, filter=True, sortable=True, resizable=True)
    gb.configure_side_bar()
    gb.configure_pagination(enabled=True, paginationPageSize=15)
    
    # Enable the header checkbox to select/deselect all rows
    gb.configure_selection(
        "multiple", 
        use_checkbox=True, 
        header_checkbox=True, 
        header_checkbox_filtered_only=False
    )

    grid_options = gb.build()

    # --- 4. Render AgGrid ---
    st.header("Uploaded Documents")
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        fit_columns_on_grid_load=False,
        height=600,
        theme="streamlit"
    )

    edited_df = pd.DataFrame(grid_response["data"])
    selected_rows = pd.DataFrame(grid_response["selected_rows"]) # Ensure selected_rows is a DataFrame

    # --- 5. CRUD Operations ---
    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save Changes", use_container_width=True):
            if not edited_df.empty:
                for _, row in edited_df.iterrows():
                    row_dict = row.to_dict()
                    row_dict.pop('chunk_count', None) 
                    supabase.table("uploaded_documents").update(row_dict).eq("id", row["id"]).execute()
                st.success("✅ All changes saved to Supabase.")
                st.cache_data.clear()
    
    with col2:
        # --- THIS IS THE FIX ---
        # A direct boolean check 'if selected_rows:' is ambiguous for a DataFrame.
        # We must explicitly check if the DataFrame is not empty.
        if not selected_rows.empty:
            if st.button(f"🗑 Delete {len(selected_rows)} Selected Row(s)", type="primary", use_container_width=True):
                # When iterating over a DataFrame of selected rows, use .iterrows()
                for _, row in selected_rows.iterrows():
                    file_name_to_delete = row['name']
                    st.info(f"Deleting chunks for {file_name_to_delete}...")
                    supabase.table("documents").delete().eq('metadata->>source', file_name_to_delete).execute()
                    
                    st.info(f"Deleting record for {file_name_to_delete}...")
                    supabase.table("uploaded_documents").delete().eq("id", row["id"]).execute()

                st.success("Rows and associated chunks deleted. The page will now refresh.")
                st.cache_data.clear()
                st.rerun()

    # --- 6. Export Options ---
    with st.expander("📤 Export Data"):
        export_format = st.selectbox("Select export format", ["CSV", "Excel"])

        if st.button("Download Data"):
            if export_format == "CSV":
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", data=csv, file_name="uploaded_documents.csv", mime="text/csv")
            else:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Uploads')
                st.download_button(
                    label="📥 Download Excel",
                    data=output.getvalue(),
                    file_name="uploaded_documents.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

except Exception as e:
    st.error("An error occurred while loading the dashboard.")
    st.exception(e)