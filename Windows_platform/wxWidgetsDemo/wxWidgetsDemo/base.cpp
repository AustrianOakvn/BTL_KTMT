#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include<wx/wx.h>
#endif
#include "base.h"
#include "Matrix.h"
#include <sstream>


BEGIN_EVENT_TABLE(MainFrame, wxFrame)
//EVT_BUTTON(BUTTON_Hello, MainFrame::OnExit)
EVT_MENU(MENU_New, MainFrame::NewFile)
//EVT_MENU(MENU_Open, MainFrame::OpenFile)
//EVT_MENU(MENU_Close, MainFrame::CloseFile)
//EVT_MENU(MENU_Save, MainFrame::SaveFile)
//EVT_MENU(MENU_SaveAs, MainFrame::SaveFileAs)
EVT_BUTTON(GENERATE_MATRIX_1, MainFrame::GenerateMatrix_1)
EVT_BUTTON(GENERATE_MATRIX_2, MainFrame::GenerateMatrix_2)
EVT_BUTTON(MATRIX_MUL, MainFrame::Matrix_Mul)
EVT_BUTTON(MATRIX_MUL, MainFrame::Matrix_Add)
EVT_MENU(MENU_Quit, MainFrame::Quit)
END_EVENT_TABLE()


IMPLEMENT_APP(MainApp)

bool MainApp::OnInit() {
	MainFrame* MainWin = new MainFrame(_("Hello World!"), wxDefaultPosition
		, wxSize(700, 600));
	//MainWin->SetSize(300, 200);
	MainWin->Show(true);
	SetTopWindow(MainWin);
	return true;
}



MainFrame::MainFrame(const wxString& title, const wxPoint& pos, const wxSize& size) : wxFrame((wxFrame*)NULL, -1, title, pos, size) {
	CreateStatusBar(2);
	//HelloWorld = new wxButton(this, BUTTON_Hello, _T("Hello World"), wxDefaultPosition, wxDefaultSize, 0);
	
	MainMenu = new wxMenuBar();
	wxMenu* FileMenu = new wxMenu();
	parent_panel = new wxPanel(this, -1);
	first_panel = new wxPanel(parent_panel, -1, wxPoint(10, 10), wxSize(230, 700));
	second_panel = new wxPanel(parent_panel, -1, wxPoint(240, 10), wxSize(130, 700));
	third_panel = new wxPanel(parent_panel, -1, wxPoint(370, 10), wxSize(270, 700));

	FileMenu->Append(MENU_New, wxT("&New"), wxT("Create a new file"));
	FileMenu->Append(MENU_Open, wxT("&Open"), wxT("Open an existing file"));
	FileMenu->Append(MENU_Close, wxT("&Close"), wxT("Close the current document"));
	FileMenu->Append(MENU_Save, wxT("&Save"), wxT("Save the current document"));
	FileMenu->Append(MENU_SaveAs, wxT("Save &As"), wxT("Save the current document under a new file name"));
	FileMenu->Append(MENU_Quit, "&Quit", "Quit the editor");
	MainMenu->Append(FileMenu, "&File");
	SetMenuBar(MainMenu);


	//MainEditBox = new wxTextCtrl(this, TEXT_Main, "Hi!", wxPoint(10, 10), wxSize(100, 100), wxTE_MULTILINE | wxTE_RICH, wxDefaultValidator, wxTextCtrlNameStr);
	wxPoint ref_point = wxPoint(10, 10);
	int margin = 10;
	matrix_1_w_label = new wxStaticText(first_panel, -1, "matrix 1 rows", ref_point, wxSize(matrix_width_label_w, matrix_heigh_label_h));
	Matrix_1_rows = new wxTextCtrl(first_panel, -1, "", wxPoint(ref_point.x+matrix_width_label_w, ref_point.y), wxSize(matrix_width_label_w, matrix_heigh_label_h));
	matrix_1_h_label = new wxStaticText(first_panel, -1, "matrix 1 cols", wxPoint(ref_point.x+matrix_width_label_w*2+margin, ref_point.y), wxSize(matrix_width_label_w, matrix_heigh_label_h));
	Matrix_1_cols = new wxTextCtrl(first_panel, -1, "", wxPoint(ref_point.x+matrix_width_label_w*3+margin, ref_point.y), wxSize(matrix_width_label_w, matrix_heigh_label_h));
	MatrixBox_1 = new wxTextCtrl(first_panel, TEXT_Matrix_1, "Input matrix here", wxPoint(ref_point.x, 50), wxSize(matrix_box_w+margin, matrix_box_h), wxTE_MULTILINE);
	random_mat_1 = new wxButton(first_panel, GENERATE_MATRIX_1, "Generate random matrix 1", wxPoint(ref_point.x, 200), wxSize(matrix_box_w+margin, 30));

	wxPoint ref_point_2 = wxPoint(10, 240);
	matrix_2_w_label = new wxStaticText(first_panel, -1, "matrix 2 rows", ref_point_2, wxSize(matrix_width_label_w, matrix_heigh_label_h));
	Matrix_2_rows = new wxTextCtrl(first_panel, -1, "", wxPoint(ref_point_2.x + matrix_width_label_w, ref_point_2.y), wxSize(matrix_width_label_w, matrix_heigh_label_h));
	matrix_2_h_label = new wxStaticText(first_panel, -1, "matrix 2 cols", wxPoint(ref_point_2.x + matrix_width_label_w * 2 + margin, ref_point_2.y), wxSize(matrix_width_label_w, matrix_heigh_label_h));
	Matrix_2_cols = new wxTextCtrl(first_panel, -1, "", wxPoint(ref_point_2.x + matrix_width_label_w * 3 + margin, ref_point_2.y), wxSize(matrix_width_label_w, matrix_heigh_label_h));
	MatrixBox_2 = new wxTextCtrl(first_panel, TEXT_Matrix_2, "Input second matrix", wxPoint(ref_point_2.x, 280), wxSize(matrix_box_w+margin, matrix_box_h), wxTE_MULTILINE);
	random_mat_2 = new wxButton(first_panel, GENERATE_MATRIX_2, "Generate random matrix 2", wxPoint(ref_point_2.x, 440), wxSize(matrix_box_w + margin, 30));
	//MainEditBox->SetSize(100, 100);

	// second panel widgets
	wxPoint ref_point_3 = wxPoint(10, 10);
	matrix_mul = new wxButton(second_panel, MATRIX_MUL, "AxB", wxPoint(ref_point_3.x, ref_point_3.y), wxSize((matrix_box_w + margin)/2, 30));
	matrix_add = new wxButton(second_panel, MATRIX_ADD, "A+B", wxPoint(ref_point_3.x, ref_point_3.y + 40), wxSize((matrix_box_w + margin)/2, 30));
	
	// third panel widgets
	wxPoint ref_point_4 = wxPoint(10, 10);
	Output_label = new wxStaticText(third_panel, -1, "Output", wxPoint(ref_point_4.x, ref_point_4.y), wxSize(matrix_box_w + margin, 30));
	Output = new wxTextCtrl(third_panel, -1, "", wxPoint(ref_point_4.x, ref_point_4.y + 30), wxSize(matrix_box_w + margin*4, 300));
	TimeOutput_label = new wxStaticText(third_panel, -1, "Calculated time in milisecond", wxPoint(ref_point_4.x, ref_point_4.y + 30 + 300), wxSize(matrix_box_w + margin, 30));
	TimeOutput = new wxTextCtrl(third_panel, -1, "", wxPoint(ref_point_4.x, ref_point_4.y + 330 + 30), wxSize(matrix_box_w + margin, 30));
	Clear = new wxButton(third_panel, CLEAR_OUTPUT, "Clear", wxPoint(ref_point_4.x + matrix_box_w+margin*4-50, ref_point_4.y + 430), wxSize(50, 30));

	//Maximize();

}

//void MainFrame::OnExit(wxCommandEvent& event) {
//	Close(TRUE);
//}

void MainFrame::NewFile(wxCommandEvent& WXUNUSED(event)) {

}

//void MainFrame::OpenFile(wxCommandEvent& WXUNUSED(event)) {
//	MainEditBox->LoadFile(wxT("base.h"));
//}

//void MainFrame::OpenFile(wxCommandEvent& WXUNUSED(event)) {
//	Matrix_size_EditBox->LoadFile(wxT("base.h"));
//}

//void MainFrame::CloseFile(wxCommandEvent& WXUNUSED(event)) {
//	MainEditBox->Clear();
//}
//
//void MainFrame::SaveFile(wxCommandEvent& WXUNUSED(event)) {
//	MainEditBox->SaveFile(wxT("base.h"));
//}

//void MainFrame::SaveFileAs(wxCommandEvent& WXUNUSED(event)) {
//
//}

void MainFrame::Quit(wxCommandEvent& WXUNUSED(event)) {
	Close(TRUE);
}

void MainFrame::GenerateMatrix_1(wxCommandEvent& WXUNUSED(event)) {
	std::string mat_1_r = Matrix_1_rows->GetValue().ToStdString();
	std::string mat_1_c = Matrix_1_cols->GetValue().ToStdString();
	if (mat_1_r.empty() || mat_1_c.empty()) {
		std::cout << "Need to specify the number of rows and columns";
	}
	else {
		matrix_1_rows = std::stoi(mat_1_r);
		matrix_1_cols = std::stoi(mat_1_c);
		matrix_1 = Matrix(matrix_1_rows, matrix_1_cols);
		matrix_1.generate_random();
		if (matrix_1_rows < 1000 && matrix_2_rows < 1000) {
			std::string mat_1_rep = matrix_1.get_presentation();
			MatrixBox_1->ChangeValue(mat_1_rep);
		}
		else{
			MatrixBox_1->ChangeValue("Matrix too large cannot display");
		}
	}
}

void MainFrame::GenerateMatrix_2(wxCommandEvent& WXUNUSED(event)) {
	std::string mat_2_r = Matrix_2_rows->GetValue().ToStdString();
	std::string mat_2_c = Matrix_2_cols->GetValue().ToStdString();
	if (mat_2_r.empty() || mat_2_c.empty()) {
		std::cout << "Need to specify the number of rows and columns";
	}
	else{
		matrix_2_rows = std::stoi(mat_2_r);
		matrix_2_cols = std::stoi(mat_2_c);
		matrix_2 = Matrix(matrix_2_rows, matrix_2_cols);
		matrix_2.generate_random();
		if (matrix_2_rows < 1000 && matrix_2_cols < 1000) {
			std::string mat_2_rep = matrix_2.get_presentation();
			MatrixBox_2->ChangeValue(mat_2_rep);
		}
		else {
			MatrixBox_2->ChangeValue("Matrix too large cannot display");
		}
	}
}

void MainFrame::Matrix_Mul(wxCommandEvent& WXUNUSED(event)) {
	Matrix result = matrix_1 * matrix_2;
	int result_rows = result.get_num_rows();
	int result_cols = result.get_num_cols();
	std::ostringstream out;
	out << matrix_2.fp_ms.count();
	std::string rep = result.get_presentation();
	std::string ret = out.str();
	if (result_rows < 1000 && result_cols < 1000) {
		Output->WriteText(rep);
		Output->WriteText("\n");
	}
	else {
		Output->WriteText("Matrix too large cannot display");
	}
	//Output->WriteText("Time measure in miliseconds");
	//Output->WriteText(ret);
	TimeOutput->ChangeValue(ret);
}

void MainFrame::Matrix_Add(wxCommandEvent& WXUNUSED(event)) {
	
}