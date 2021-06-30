#pragma once
//#include <wx/wxprec.h>
#include "Matrix.h"

class MainApp :public wxApp {
public:
	virtual bool OnInit();
};


class MainFrame :public wxFrame {
private:
	// matrix
	Matrix matrix_1;
	Matrix matrix_2;
	// matrix 1 size
	int matrix_1_rows, matrix_1_cols;
	int matrix_2_rows, matrix_2_cols;
		
	// windows widgets size param
	int first_panel_w = 300;
	int first_panel_h = 500;
	int matrix_width_label_w = 50;
	int matrix_heigh_label_h = 30;
	int matrix_box_w = 200;
	int matrix_box_h = 150;

public:
	MainFrame(const wxString& title, const wxPoint& pos, const wxSize& size);
	
	//wxButton* HelloWorld;
	wxTextCtrl* Matrix_1_rows;
	wxTextCtrl* Matrix_1_cols;
	wxTextCtrl* Matrix_2_rows;
	wxTextCtrl* Matrix_2_cols;

	wxStaticText* matrix_1_w_label;
	wxStaticText* matrix_1_h_label;
	wxStaticText* matrix_2_w_label;
	wxStaticText* matrix_2_h_label;

	wxTextCtrl* MatrixBox_1;
	wxTextCtrl* MatrixBox_2;
	wxMenuBar* MainMenu;


	wxPanel* parent_panel;
	wxPanel* first_panel;
	wxPanel* second_panel;
	wxPanel* third_panel;

	wxStaticText* matrix_label_1;
	wxStaticText* matrix_label_2;
	wxButton* random_mat_1;
	wxButton* random_mat_2;

	// second panel
	wxButton* matrix_mul;
	wxButton* matrix_add;

	// third panel
	wxStaticText* Output_label;
	wxTextCtrl* Output;
	wxStaticText* TimeOutput_label;
	wxTextCtrl* TimeOutput;
	wxButton* Clear;

	void Quit(wxCommandEvent& event);
	void NewFile(wxCommandEvent& event);
	//void OpenFile(wxCommandEvent& event);
	//void SaveFile(wxCommandEvent& event);
	//void SaveFileAs(wxCommandEvent& event);
	//void CloseFile(wxCommandEvent& event);
	//void OnExit(wxCommandEvent& event);
	void GenerateMatrix_1(wxCommandEvent& event);
	void GenerateMatrix_2(wxCommandEvent& event);
	void Matrix_Mul(wxCommandEvent& event);
	void Matrix_Add(wxCommandEvent& event);

	DECLARE_EVENT_TABLE()
};

enum {
	//BUTTON_Hello = wxID_HIGHEST + 1
	//TEXT_Main = wxID_HIGHEST + 1,
	TEXT_Matrix_1,
	TEXT_Matrix_2,
	GENERATE_MATRIX_1,
	GENERATE_MATRIX_2,
	MATRIX_MUL,
	MATRIX_ADD,
	CLEAR_OUTPUT,
	TEXT_OUTPUT,
	MENU_New,
	MENU_Open,
	MENU_Close,
	MENU_Save,
	MENU_SaveAs,
	MENU_Quit
};