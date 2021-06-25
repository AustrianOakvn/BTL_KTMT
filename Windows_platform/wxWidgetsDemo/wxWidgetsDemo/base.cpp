#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include<wx/wx.h>
#endif
#include "base.h"


BEGIN_EVENT_TABLE(MainFrame, wxFrame)
//EVT_BUTTON(BUTTON_Hello, MainFrame::OnExit)
EVT_MENU(MENU_New, MainFrame::NewFile)
EVT_MENU(MENU_Open, MainFrame::OpenFile)
EVT_MENU(MENU_Close, MainFrame::CloseFile)
EVT_MENU(MENU_Save, MainFrame::SaveFile)
EVT_MENU(MENU_SaveAs, MainFrame::SaveFileAs)
EVT_MENU(MENU_Quit, MainFrame::Quit)
END_EVENT_TABLE()


IMPLEMENT_APP(MainApp)

bool MainApp::OnInit() {
	MainFrame* MainWin = new MainFrame(_("Hello World!"), wxDefaultPosition
		, wxSize(300, 200));
	MainWin->Show(true);
	SetTopWindow(MainWin);
	return true;
}



MainFrame::MainFrame(const wxString& title, const wxPoint& pos, const wxSize& size) : wxFrame((wxFrame*)NULL, -1, title, pos, size) {
	CreateStatusBar(2);
	//HelloWorld = new wxButton(this, BUTTON_Hello, _T("Hello World"), wxDefaultPosition, wxDefaultSize, 0);
	
	MainMenu = new wxMenuBar();
	wxMenu* FileMenu = new wxMenu();

	FileMenu->Append(MENU_New, wxT("&New"), wxT("Create a new file"));
	FileMenu->Append(MENU_Open, wxT("&Open"), wxT("Open an existing file"));
	FileMenu->Append(MENU_Close, wxT("&Close"), wxT("Close the current document"));
	FileMenu->Append(MENU_Save, wxT("&Save"), wxT("Save the current document"));
	FileMenu->Append(MENU_SaveAs, wxT("Save &As"), wxT("Save the current document under a new file name"));
	FileMenu->Append(MENU_Quit, "&Quit", "Quit the editor");
	MainMenu->Append(FileMenu, "&File");
	SetMenuBar(MainMenu);


	MainEditBox = new wxTextCtrl(this, TEXT_Main, "Hi!", wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE | wxTE_RICH, wxDefaultValidator, wxTextCtrlNameStr);
	Maximize();

}

//void MainFrame::OnExit(wxCommandEvent& event) {
//	Close(TRUE);
//}

void MainFrame::NewFile(wxCommandEvent& WXUNUSED(event)) {

}

void MainFrame::OpenFile(wxCommandEvent& WXUNUSED(event)) {
	MainEditBox->LoadFile(wxT("base.h"));
}

void MainFrame::CloseFile(wxCommandEvent& WXUNUSED(event)) {
	MainEditBox->Clear();
}

void MainFrame::SaveFile(wxCommandEvent& WXUNUSED(event)) {
	MainEditBox->SaveFile(wxT("base.h"));
}

void MainFrame::SaveFileAs(wxCommandEvent& WXUNUSED(event)) {

}

void MainFrame::Quit(wxCommandEvent& WXUNUSED(event)) {
	Close(TRUE);
}